"""Runner: controlled training, evaluation, telemetry, cancellation, checkpoints.

One active worker. Every job is charged to a cumulative budget before it is
allowed to start. A job that stops on its time cap is `CENSORED`, not a
completed equal-data result, and no automatic retries or smaller batches exist.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .datasets import mqar
from .datasets.manifest import load_examples, load_manifest
from .models import TorchMQARAdapter, encode_batch, registered_adapters
from .scoring import aggregate, score_item
from .util import append_jsonl, environment_record, sha256_file, write_json

FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)


@dataclass
class Limits:
    max_gpu_fraction: float = 0.80
    max_process_ram_fraction: float = 0.50
    max_new_artifacts_bytes: int = 10 * 1024**3
    max_checkpoints_per_job: int = 3
    eval_example_timeout_s: float = 30.0
    eval_per_cell: int = 512
    eval_batch_size: int = 64
    resource_sample_interval_s: float = 5.0
    grad_accum_steps: int = 1

    def to_dict(self) -> dict:
        return {
            "max_gpu_fraction": self.max_gpu_fraction,
            "max_process_ram_fraction": self.max_process_ram_fraction,
            "max_new_artifacts_bytes": self.max_new_artifacts_bytes,
            "max_checkpoints_per_job": self.max_checkpoints_per_job,
            "eval_example_timeout_s": self.eval_example_timeout_s,
            "eval_per_cell": self.eval_per_cell,
            "eval_batch_size": self.eval_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
        }


class Control:
    """Cooperative pause/cancel observed from a control file and/or SIGINT."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.cancel_requested = False
        self.pause_requested = False

    def poll(self) -> None:
        if self.path and self.path.exists():
            try:
                doc = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return
            self.cancel_requested = self.cancel_requested or bool(doc.get("cancel"))
            self.pause_requested = bool(doc.get("pause"))

    def clear(self) -> None:
        if self.path and self.path.exists():
            write_json(self.path, {"cancel": False, "pause": False})
        self.cancel_requested = False
        self.pause_requested = False


def _gpu_memory() -> tuple[int | None, int | str]:
    """Free/total accelerator memory in bytes, tolerating an uninitialisable context."""
    try:
        import torch

        free, total = torch.cuda.mem_get_info(0)
        return int(free), int(total)
    except Exception as exc:
        # A context that cannot even initialise still tells us free memory via the driver.
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            ).stdout.strip().splitlines()[0]
            free_mib, total_mib = (int(x) for x in out.split(","))
            return free_mib * 1024**2, total_mib * 1024**2
        except Exception:
            return None, str(exc).splitlines()[0]


def resource_preflight(device: str, *, params: int, ctx_tokens: int, batch: int) -> tuple[bool, str]:
    """Refuse to start if the selected device cannot plausibly host the job.

    The spec forbids silently shrinking the model or context; a failed preflight
    is an explicit protocol revision, so this returns a reason instead.
    """
    if device != "cuda":
        return True, ""
    free, total = _gpu_memory()
    if free is None:
        return False, f"cannot query accelerator memory: {total}"
    # weights + AdamW moments + grads (fp32) plus a generous activation allowance
    weights = params * 4 * 4
    activations = batch * ctx_tokens * 4 * 64 * 8
    needed = int(weights + activations + 512 * 1024**2)
    if free < needed:
        return False, (
            f"insufficient free accelerator memory: {free / 1024**2:.0f} MiB free of "
            f"{total / 1024**2:.0f} MiB, need ~{needed / 1024**2:.0f} MiB for "
            f"{params:,} params at batch {batch} x {ctx_tokens} tokens"
        )
    return True, ""


@dataclass
class JobResult:
    job_id: str
    status: str
    summary: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


def lr_at(step: int, total: int, base_lr: float, warmup_fraction: float = 0.05) -> float:
    """Linear warm-up then cosine decay over the declared budget."""
    if total <= 0:
        return base_lr
    warmup = max(1, int(round(warmup_fraction * total)))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _total_ram_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


class StudyRunner:
    def __init__(
        self,
        study_dir: str | Path,
        resolved: dict,
        *,
        device: str,
        limits: Limits | None = None,
        control: Control | None = None,
        log=print,
    ) -> None:
        self.study_dir = Path(study_dir)
        self.resolved = resolved
        self.device = device
        self.limits = limits or Limits(eval_per_cell=int(resolved.get("eval_per_cell", 512)))
        self.control = control or Control(self.study_dir / "control.json")
        self.log = log
        self.consumed_s = 0.0
        self.skipped_jobs: list[dict] = []
        self._artifacts_start = 0
        self.gates: dict = {}

    # ---------------------------------------------------------------- setup

    def preflight(self) -> tuple[bool, list[str]]:
        """Study-level resource preflight over the whole matrix."""
        problems: list[str] = []
        batch = int(self.resolved["effective_batch_examples"])
        ctx = max(mqar.parse_cell(c)[0] for c in self.resolved["evaluation_cells"])
        seen = set()
        for mid, row in self.resolved["capability_matrix"].items():
            params = int(row["size"]["params"])
            if params in seen:
                continue
            seen.add(params)
            ok, reason = resource_preflight(self.device, params=params, ctx_tokens=ctx, batch=batch)
            if not ok:
                problems.append(f"{mid}: {reason}")
        return not problems, problems

    def prepare(self, approval: dict) -> None:
        d = self.study_dir
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "approval-and-limits.json", {"approval": approval, "limits": self.limits.to_dict()})
        write_json(d / "environment.json", environment_record(self.device))
        manifest_src = Path(self.resolved["dataset_manifest"])
        if not manifest_src.is_absolute():
            manifest_src = Path.cwd() / manifest_src
        if manifest_src.exists():
            shutil.copyfile(manifest_src, d / "dataset-manifest.json")
        (d / "jobs").mkdir(exist_ok=True)
        (d / "tables").mkdir(exist_ok=True)
        (d / "charts").mkdir(exist_ok=True)
        self._artifacts_start = self._dir_bytes()
        write_json(d / "control.json", {"cancel": False, "pause": False})

    def _dir_bytes(self) -> int:
        total = 0
        for root, _dirs, files in os.walk(self.study_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    # ---------------------------------------------------------------- data

    def _manifest(self) -> dict:
        return load_manifest(self.resolved["dataset_manifest"] if Path(self.resolved["dataset_manifest"]).is_absolute()
                             else Path.cwd() / self.resolved["dataset_manifest"])

    def _heldout(self) -> tuple[list[mqar.Example], list[mqar.Example]]:
        doc = self._manifest()
        base = Path(self.resolved["dataset_manifest"])
        base = base.parent if base.is_absolute() else Path.cwd() / base.parent
        val: list[mqar.Example] = []
        test: list[mqar.Example] = []
        for cid in self.resolved["evaluation_cells"]:
            for split, bucket in (("validation", val), ("test", test)):
                rec = doc["splits"][split][cid]
                path = base / rec["path"]
                if sha256_file(path) != rec["sha256"]:
                    raise RuntimeError(f"split hash mismatch for {split}/{cid}")
                examples = load_examples(path)
                bucket.extend(examples[: self.limits.eval_per_cell])
        return val, test

    # ---------------------------------------------------------------- eval

    def _evaluate(
        self,
        adapter: TorchMQARAdapter,
        examples: list[mqar.Example],
        *,
        split: str,
        predictions_path: Path | None = None,
    ) -> dict:
        results = []
        by_cell: dict[str, list[mqar.Example]] = {}
        for ex in examples:
            by_cell.setdefault(ex.cell, []).append(ex)
        bs = self.limits.eval_batch_size
        for cell, group in by_cell.items():
            group = sorted(group, key=lambda e: (e.length, e.index))
            for start in range(0, len(group), bs):
                chunk = group[start : start + bs]
                batch = encode_batch(chunk, self.device)
                t0 = time.perf_counter()
                try:
                    logits = adapter.predict(batch)
                except Exception as exc:  # evaluation failure stays in the denominator
                    for ex in chunk:
                        results.append(score_item(ex, None, failure=f"predict_error:{type(exc).__name__}"))
                    continue
                dt = (time.perf_counter() - t0) / max(1, len(chunk))
                if dt > self.limits.eval_example_timeout_s:
                    for ex in chunk:
                        results.append(score_item(ex, None, failure="evaluation_timeout"))
                    continue
                for ex, lg in zip(chunk, logits):
                    results.append(score_item(ex, lg.numpy()))
        agg = aggregate(results, planned=len(examples))
        agg["split"] = split
        if predictions_path is not None:
            append_jsonl(
                predictions_path,
                [
                    {
                        "split": split,
                        "example_id": r.example_id,
                        "cell": r.cell,
                        "n_queries": r.n_queries,
                        "correct": r.correct,
                        "nll_sum_nats": r.nll_sum,
                        "predicted": r.predicted,
                        "targets": r.targets,
                        "failure": r.failure,
                    }
                    for r in results
                ],
            )
        return agg

    # ---------------------------------------------------------------- job

    def _early_summary(self, job: dict, status: str, reason: str, extra: dict | None = None) -> JobResult:
        """A job that never started is still a recorded outcome, not a silence."""
        job_dir = self.study_dir / "jobs" / job["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "job_id": job["job_id"],
            "status": status,
            "stop_reason": reason,
            "model_id": job["model_id"],
            "stage": job["stage"],
            "seed": job["seed"],
            "learning_rate": job.get("learning_rate"),
            "updates_completed": 0,
            "updates_planned": job["updates"],
            "censored": False,
            "elapsed_s": 0.0,
            "resumable_checkpoint": None,
            "selected_checkpoint": None,
            "protocol_sha256": self.resolved.get("protocol_sha256"),
            "device": self.device,
            **(extra or {}),
        }
        write_json(job_dir / "summary.json", summary)
        append_jsonl(job_dir / "events.jsonl", [{"event": "finished", "status": status, "stop_reason": reason}])
        return JobResult(job["job_id"], status, summary, 0.0)

    def run_job(self, job: dict, *, resume_from: str | Path | None = None) -> JobResult:
        self.control.poll()
        if self.control.cancel_requested:
            return self._early_summary(job, "CANCELLED", "cancelled before start")

        ceiling = float(self.resolved["total_accelerator_hours_limit"])
        if self.consumed_s / 3600.0 + job["time_cap_s"] / 3600.0 > ceiling:
            return self._early_summary(
                job,
                "BUDGET_EXHAUSTED",
                f"job allocation {job['time_cap_s'] / 3600:.3f}h exceeds remaining approved budget",
            )

        mid = job["model_id"]
        model_row = self.resolved["capability_matrix"][mid]
        cell_status = {c: v["status"] for c, v in model_row["cells"].items()}
        if set(cell_status.values()) == {"UNSUPPORTED"}:
            return self._early_summary(job, "UNSUPPORTED", "no supported cells", {"cells": model_row["cells"]})

        job_dir = self.study_dir / "jobs" / job["job_id"]
        (job_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        size = model_row["size"]
        config = {
            "width": size["width"],
            "depth": size["depth"],
            "n_heads": size["heads"],
            "target_trainable_parameters": size["target"],
            "parameter_tolerance_fraction": size["tolerance"],
            "parameter_matched_claimed": size["matched"],
            "parameter_deviation": size["deviation"],
            "precision": self.resolved.get("precision", "float32"),
            "effective_batch_examples": self.resolved["effective_batch_examples"],
        }
        adapters = registered_adapters(
            source_commit=self.resolved["adapter_commits"].get(mid, "unknown"), device=self.device
        )
        adapter = adapters[mid]
        adapter.build(config, initialization_seed=job["seed"])
        write_json(job_dir / "model-effective-config.json", adapter.metadata())

        optimizer = torch.optim.AdamW(
            adapter.model.parameters(), lr=0.0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
        )
        total_updates = int(job["updates"])
        base_lr = float(job.get("learning_rate") or self.resolved["learning_rates"][0])

        manifest = self._manifest()
        heldout_hash_set: set[str] = set()
        base_manifest = Path(self.resolved["dataset_manifest"])
        base_manifest = base_manifest.parent if base_manifest.is_absolute() else Path.cwd() / base_manifest.parent
        for split in ("validation", "test"):
            for cid, rec in manifest["splits"][split].items():
                for ex in load_examples(base_manifest / rec["path"]):
                    heldout_hash_set.add(ex.content_hash())

        stream = mqar.TrainingStream(
            cells=[mqar.parse_cell(c) for c in manifest["train"]["cells"]],
            generator_seed=int(mqar_manifest_seed(manifest)),
            heldout_hashes=heldout_hash_set,
        )
        step = 0
        resumed = False
        if resume_from:
            payload = adapter.load_checkpoint(resume_from)
            extra = payload.get("extra", {})
            opt_state = extra.get("optimizer")
            if opt_state:
                optimizer.load_state_dict(opt_state)
            if "stream" in extra:
                stream.load_state(extra["stream"])
            step = int(extra.get("step", 0))
            resumed = True
            append_jsonl(job_dir / "events.jsonl", [{"event": "resumed", "step": step, "checkpoint": str(resume_from)}])

        append_jsonl(
            job_dir / "events.jsonl",
            [
                {
                    "event": "started",
                    "job": job,
                    "adapter": adapter.metadata(),
                    "resumed": resumed,
                    "protocol_sha256": self.resolved["protocol_sha256"],
                    "device": self.device,
                    "timestamp": time.time(),
                }
            ],
            fsync=True,
        )

        evals: list[dict] = []
        next_eval_idx = 0
        # 100% is evaluated after the loop, so a 1.0 checkpoint is never made twice.
        fraction_steps = [max(1, int(round(f * total_updates))) for f in FRACTIONS[:-1]]
        while next_eval_idx < len(fraction_steps) and fraction_steps[next_eval_idx] <= step:
            next_eval_idx += 1

        best = {"score": -1.0, "step": None, "path": None}
        final_path = None
        status = "COMPLETED"
        stop_reason = None
        train_start = time.perf_counter()
        last_res_sample = 0.0
        metrics_batch: list[dict] = []

        while step < total_updates:
            self.control.poll()
            if self.control.cancel_requested:
                status, stop_reason = "CANCELLED", "cancel_requested"
                break
            if self.control.pause_requested:
                status, stop_reason = "PAUSED", "pause_requested"
                break
            elapsed = time.perf_counter() - train_start
            if elapsed > job["time_cap_s"]:
                status, stop_reason = "CENSORED", "time_cap"
                break

            batch_examples = stream.batch(int(self.resolved["effective_batch_examples"]))
            batch = encode_batch(batch_examples, self.device)
            lr = lr_at(step, total_updates, base_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            out = adapter.train_step(batch, optimizer)
            step += 1
            now = time.perf_counter()
            metrics_batch.append(
                {
                    "step": step,
                    "loss": out["loss"],
                    "grad_norm_preclip": out["grad_norm_preclip"],
                    "lr": lr,
                    "elapsed_s": now - train_start,
                    "tokens": adapter.counters["tokens"],
                    "supervised_targets": adapter.counters["supervised_targets"],
                    "examples_seen": adapter.counters["examples"],
                    "stream_counter": stream.counter,
                    "stream_skipped": stream.skipped,
                    "device": self.device,
                }
            )
            if len(metrics_batch) >= 10 or step == total_updates:
                append_jsonl(job_dir / "training-metrics.jsonl", metrics_batch)
                metrics_batch = []

            if now - last_res_sample >= self.limits.resource_sample_interval_s:
                last_res_sample = now
                rec = {
                    "step": step,
                    "elapsed_s": now - train_start,
                    "rss_bytes": _rss_bytes(),
                    "rss_cap_bytes": int(_total_ram_bytes() * self.limits.max_process_ram_fraction),
                }
                if self.device == "cuda":
                    rec["gpu_alloc_bytes"] = torch.cuda.memory_allocated()
                    rec["gpu_reserved_bytes"] = torch.cuda.memory_reserved()
                    rec["gpu_peak_alloc_bytes"] = torch.cuda.max_memory_allocated()
                    rec["gpu_cap_bytes"] = int(
                        torch.cuda.get_device_properties(0).total_memory * self.limits.max_gpu_fraction
                    )
                append_jsonl(job_dir / "resource-metrics.jsonl", [rec])
                if rec["rss_bytes"] > rec["rss_cap_bytes"]:
                    status, stop_reason = "FAILED", "process_ram_cap_exceeded"
                    break

            if next_eval_idx < len(fraction_steps) and step >= fraction_steps[next_eval_idx]:
                val_examples, _ = self._heldout()
                val = self._evaluate(
                    adapter,
                    val_examples,
                    split="validation",
                    predictions_path=job_dir / "evaluation-predictions.jsonl",
                )
                extra = {
                    "optimizer": optimizer.state_dict(),
                    "stream": stream.state(),
                    "step": step,
                    "fraction": FRACTIONS[next_eval_idx],
                }
                ckpt = job_dir / "checkpoints" / f"step{step:06d}.pt"
                adapter.save_checkpoint(ckpt, extra=extra)
                evals.append({"step": step, "fraction": FRACTIONS[next_eval_idx], "checkpoint": str(ckpt), "validation": val})
                write_json(job_dir / "evaluation-validation.json", val)
                append_jsonl(
                    job_dir / "events.jsonl",
                    [{"event": "validation", "step": step, "fraction": FRACTIONS[next_eval_idx],
                      "cell_macro_query_accuracy": val["cell_macro_query_accuracy"], "loss": val["cross_entropy_nats"] if "cross_entropy_nats" in val else None}],
                )
                score = val["cell_macro_query_accuracy"]
                if score > best["score"]:
                    best = {"score": score, "step": step, "path": str(ckpt)}
                next_eval_idx += 1
                self._prune_checkpoints(job_dir, keep={best["path"]}, latest=ckpt)
                final_path = ckpt

        # ---- end of training: always leave a resumable checkpoint
        resumable = None
        test_summary = None
        ckpt = job_dir / "checkpoints" / f"step{step:06d}.pt"
        adapter.save_checkpoint(
            ckpt, extra={"optimizer": optimizer.state_dict(), "stream": stream.state(), "step": step}
        )
        final_path = ckpt
        resumable = ckpt

        # validation + test evaluation only for a job that reached its allocation
        if status in ("COMPLETED", "CENSORED"):
            val_examples, test_examples = self._heldout()
            final_val = self._evaluate(
                adapter, val_examples, split="validation", predictions_path=job_dir / "evaluation-predictions.jsonl"
            )
            if best["step"] is None or final_val["cell_macro_query_accuracy"] > best["score"]:
                best = {"score": final_val["cell_macro_query_accuracy"], "step": step, "path": str(ckpt)}
            evals.append({"step": step, "fraction": step / total_updates, "checkpoint": str(ckpt), "validation": final_val})
            self._prune_checkpoints(job_dir, keep={best["path"]}, latest=ckpt)

            # Test evaluation happens only after training and selection finish.
            if job["stage"] == "confirmation" and status == "COMPLETED":
                adapter.load_checkpoint(best["path"])
                test_summary = self._evaluate(adapter, test_examples, split="test")
                write_json(job_dir / "evaluation-test.json", test_summary)

        elapsed_total = time.perf_counter() - train_start
        self.consumed_s += elapsed_total
        summary = {
            "job_id": job["job_id"],
            "status": status,
            "stop_reason": stop_reason,
            "model_id": mid,
            "stage": job["stage"],
            "seed": job["seed"],
            "learning_rate": base_lr,
            "updates_completed": step,
            "updates_planned": total_updates,
            "censored": status == "CENSORED",
            "elapsed_s": elapsed_total,
            "size": size,
            "counters": adapter.counters,
            "stream": {"counter": stream.counter, "skipped": stream.skipped},
            "validation_curve": [
                {"step": e["step"], "fraction": e["fraction"],
                 "cell_macro_query_accuracy": e["validation"]["cell_macro_query_accuracy"],
                 "query_accuracy": e["validation"]["query_accuracy"],
                 "sequence_accuracy": e["validation"]["cell_macro_sequence_accuracy"]}
                for e in evals
            ],
            "final_validation": evals[-1]["validation"] if evals else None,
            "selected_checkpoint": best["path"],
            "selected_step": best["step"],
            "final_checkpoint": final_path,
            "resumable_checkpoint": str(resumable) if resumable else None,
            "test": test_summary if status == "COMPLETED" and job["stage"] == "confirmation" else None,
            "device": self.device,
            "precision": self.resolved.get("precision", "float32"),
            "protocol_sha256": self.resolved["protocol_sha256"],
            "checkpoint_sha256": sha256_file(best["path"]) if best["path"] and Path(best["path"]).exists() else None,
            "flop_estimate": {
                "status": "incomplete",
                "note": "6*params*tokens only; attention and memory terms are not included",
                "value": 6 * int(size["params"]) * int(adapter.counters["tokens"]),
            },
        }
        write_json(job_dir / "summary.json", summary)
        append_jsonl(job_dir / "events.jsonl", [{"event": "finished", "status": status, "stop_reason": stop_reason,
                                                "step": step, "elapsed_s": elapsed_total}], fsync=True)

        self._enforce_artifact_cap()
        return JobResult(job["job_id"], status, summary, elapsed_total)

    def _prune_checkpoints(self, job_dir: Path, *, keep: set[str | None], latest: Path | None) -> None:
        ckpt_dir = job_dir / "checkpoints"
        files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        keep_paths = {Path(k) for k in keep if k}
        if latest:
            keep_paths.add(latest)
        while len(files) > self.limits.max_checkpoints_per_job:
            victim = next((f for f in files if f not in keep_paths), None)
            if victim is None:
                break
            victim.unlink(missing_ok=True)
            files.remove(victim)

    def _enforce_artifact_cap(self) -> None:
        grown = self._dir_bytes() - self._artifacts_start
        if grown > self.limits.max_new_artifacts_bytes:
            raise RuntimeError(
                f"study artifacts grew to {grown} bytes, above the cap {self.limits.max_new_artifacts_bytes}"
            )


def mqar_manifest_seed(manifest: dict) -> int:
    return int(manifest["generator_seed"])


def run_study(
    study_dir: str | Path,
    resolved: dict,
    jobs: list[dict],
    approval: dict,
    *,
    device: str,
    limits: Limits | None = None,
    log=print,
) -> dict:
    runner = StudyRunner(study_dir, resolved, device=device, limits=limits, log=log)
    runner.prepare(approval)
    ok, problems = runner.preflight()
    write_json(
        Path(study_dir) / "preflight.json",
        {"device": device, "ok": ok, "problems": problems},
    )
    if not ok:
        for p in problems:
            log(f"preflight FAILED: {p}")
        log("refusing to start: a failed resource preflight is an explicit protocol revision, "
            "not a reason to shrink the model or context")
        return {"results": [], "consumed_s": 0.0, "preflight_ok": False, "problems": problems}
    results = []
    for job in jobs:
        res = runner.run_job(job)
        results.append(res)
        log(f"[{res.status:16s}] {res.job_id}  {res.elapsed_s:7.1f}s")
        if res.status in ("CANCELLED", "PAUSED", "BUDGET_EXHAUSTED", "FAILED"):
            break
    write_json(
        Path(study_dir) / "tables" / "jobs.json",
        [
            {
                "job_id": r.job_id,
                "status": r.status,
                "elapsed_s": r.elapsed_s,
                "model_id": r.summary.get("model_id"),
                "stage": r.summary.get("stage"),
                "seed": r.summary.get("seed"),
                "cell_macro_query_accuracy": (r.summary.get("final_validation") or {}).get("cell_macro_query_accuracy"),
            }
            for r in results
        ],
    )
    return {"results": [r.__dict__ for r in results], "consumed_s": runner.consumed_s, "preflight_ok": True}
