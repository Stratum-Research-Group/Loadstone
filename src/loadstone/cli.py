"""Command interface (spec section 17): validate, plan, smoke, freeze, run, resume, report.

Every command that could consume compute prints the expanded job matrix and the
worst-case allocation before it does anything, and requires an explicit local
confirmation flag.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .datasets import mqar
from .datasets.manifest import build_manifest, load_manifest, verify_manifest
from .protocol import (
    approval_record,
    expand_jobs,
    freeze,
    load_protocol,
    read_approval,
    resolve,
    summarize_plan,
    validate_structure,
)
from .report import build_report
from .runner import Control, Limits, StudyRunner, run_study
from .store import index_study, list_studies
from .util import environment_record, read_json, resolve_device, utc_now, write_json

REPO = Path(__file__).resolve().parents[2]
GATE_FILES = {
    "unit": [REPO / "tests" / "test_gate0_datasets.py", REPO / "tests" / "test_gate0_scoring.py"],
    "adapter": [REPO / "tests" / "test_gate1_adapter.py"],
}


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _limits_from_file(path: str | Path | None) -> Limits:
    if not path:
        return Limits()
    doc = read_json(path)
    allowed = {
        "max_gpu_fraction", "max_process_ram_fraction", "max_new_artifacts_bytes",
        "max_checkpoints_per_job", "eval_example_timeout_s", "eval_per_cell",
        "eval_batch_size", "grad_accum_steps",
    }
    unknown = set(doc) - allowed
    if unknown:
        _die(f"unknown limit fields: {sorted(unknown)}")
    return Limits(**{k: v for k, v in doc.items() if k in allowed})


def _resolve_for(protocol_path: str, device: str | None) -> dict:
    protocol = load_protocol(protocol_path)
    return resolve(protocol, base_dir=Path.cwd(), device=device)


# ------------------------------------------------------------------ commands


def cmd_data_build(args) -> int:
    out = Path(args.out)
    doc = build_manifest(
        out,
        per_cell=args.per_cell,
        include_ood=args.include_ood,
        generator_seed=args.generator_seed,
    )
    ok, problems = verify_manifest(out / "dataset-manifest.json")
    print(f"built {doc['suite']} {doc['generator_version']} -> {out}")
    print(f"cells: {len(doc['cells'])}  validation/test per cell: {args.per_cell}")
    print(f"heldout sha256: {doc['heldout_sha256']}")
    print(f"verify: {'OK' if ok else problems}")
    return 0 if ok else 1


def cmd_validate(args) -> int:
    files = [str(f) for f in GATE_FILES[args.stage]]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        _die(f"missing gate test files: {missing}")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q"], cwd=str(REPO), capture_output=True, text=True
    )
    rec = {
        "stage": args.stage,
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "files": [str(Path(f).relative_to(REPO)) for f in files],
        "duration_s": round(time.time() - t0, 2),
        "timestamp": utc_now(),
        "tail": (proc.stdout or proc.stderr)[-4000:],
        "environment": environment_record(resolve_device(args.device)),
    }
    if args.out:
        write_json(args.out, rec)
    print((proc.stdout or proc.stderr).strip())
    print(f"gate {args.stage}: {'PASS' if rec['passed'] else 'FAIL'}")
    return 0 if rec["passed"] else 1


def cmd_plan(args) -> int:
    resolved = _resolve_for(args.protocol, args.device)
    jobs = resolved["job_matrix"]
    print(summarize_plan(resolved))
    if args.expand_jobs:
        print("\nexpanded job matrix:")
        for j in jobs:
            print(
                f"  {j['job_id']:34s} stage={j['stage']:12s} updates={j['updates']:>6d} "
                f"time_cap={j['time_cap_s']:>5d}s"
            )
    unsupported = [
        (mid, cell, v["status"])
        for mid, row in resolved["capability_matrix"].items()
        for cell, v in row["cells"].items()
        if v["status"] != "SUPPORTED"
    ]
    print(f"\nunsupported/gate-failed cells: {len(unsupported)}")
    for mid, cell, status in unsupported[:20]:
        print(f"  {mid} {cell}: {status}")
    if args.out:
        write_json(args.out, resolved)
    return 0


def _run_gates(study_dir: Path, device: str | None) -> dict:
    """Run Gate 0 and Gate 1 and persist the evidence next to the study."""
    gates = {}
    for stage in ("unit", "adapter"):
        files = [str(f) for f in GATE_FILES[stage]]
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *files, "-q"], cwd=str(REPO), capture_output=True, text=True
        )
        gates[stage] = {
            "stage": stage,
            "passed": proc.returncode == 0,
            "returncode": proc.returncode,
            "files": [str(Path(f).relative_to(REPO)) for f in files],
            "duration_s": round(time.time() - t0, 2),
            "timestamp": utc_now(),
            "tail": (proc.stdout or proc.stderr)[-4000:],
        }
        print(f"gate {stage}: {'PASS' if gates[stage]['passed'] else 'FAIL'}")
    gates["environment"] = environment_record(resolve_device(device))
    write_json(study_dir / "gates.json", gates)
    return gates


def cmd_smoke(args) -> int:
    resolved = _resolve_for(args.protocol, args.device)
    if args.models:
        keep = {m.strip() for m in args.models.split(",") if m.strip()}
        resolved["model_ids"] = [m for m in resolved["model_ids"] if m in keep]
    resolved["stages"] = ["smoke"]
    resolved["eval_per_cell"] = args.eval_per_cell
    jobs = expand_jobs(resolved)
    print(summarize_plan(resolved))
    print("\nSMOKE jobs (plumbing only, not a ranking):")
    for j in jobs:
        print(f"  {j['job_id']:34s} updates={j['updates']} time_cap={j['time_cap_s']}s")
    limit = _limits_from_file(args.limits)
    limit.eval_per_cell = args.eval_per_cell
    worst = sum(j["time_cap_s"] for j in jobs) / 3600
    print(f"\nworst case: {worst:.3f} accelerator-hours; eval per cell: {args.eval_per_cell}")
    if not args.yes:
        print("\nrefusing to start: re-run with --yes after reviewing the allocation above")
        return 3
    resolved["total_accelerator_hours_limit"] = args.hours or max(worst, 0.05)
    resolved["study_id"] = args.study
    study_dir = Path(args.studies_dir) / args.study
    study_dir.mkdir(parents=True, exist_ok=True)
    gates = _run_gates(study_dir, args.device)
    failing = [g for g in ("unit", "adapter") if not gates[g]["passed"]]
    if failing:
        _die(f"gates failing: {failing}; smoke cannot start while a gate fails")
    resolved["status"] = "frozen"
    from .util import sha256_json

    resolved.pop("protocol_sha256", None)
    resolved["protocol_sha256"] = sha256_json(resolved)
    write_json(study_dir / "protocol.resolved.json", resolved)
    (study_dir / "protocol.sha256").write_text(resolved["protocol_sha256"] + "\n", encoding="utf-8")
    approval = approval_record(resolved, plan_shown=True, limits=limit.to_dict())
    write_json(study_dir / "approval.json", approval)
    out = run_study(study_dir, resolved, jobs, approval, device=resolve_device(args.device), limits=limit)
    if not out.get("preflight_ok", True):
        print("\nstudy not started: resource preflight failed", file=sys.stderr)
        for p in out.get("problems", []):
            print(f"  {p}", file=sys.stderr)
        return 4
    print(f"\nconsumed {out['consumed_s']/60:.2f} minutes")
    print(build_report(study_dir))
    index_study(study_dir, Path(args.studies_dir) / "loadstone.db")
    return 0


def cmd_freeze(args) -> int:
    protocol = load_protocol(args.protocol)
    if args.hours is not None:
        protocol["total_accelerator_hours_limit"] = args.hours
    # resolve first (it pins generator version, manifest hashes and adapter
    # commits), then require the resolved document to be placeholder-free
    resolved = resolve(protocol, base_dir=Path.cwd(), device=args.device)
    problems = validate_structure(resolved, strict=True)
    if problems:
        _die("; ".join(problems))
    resolved["study_id"] = args.study
    study_dir = Path(args.studies_dir) / args.study
    study_dir.mkdir(parents=True, exist_ok=True)
    path = freeze(resolved, study_dir)
    resolved = read_json(path)
    write_json(study_dir / "plan.json", resolved["job_matrix"])
    print(summarize_plan(resolved))
    print(f"\nfrozen: {path}\nsha256: {resolved['protocol_sha256']}")
    print("\nnext: review the plan, then `loadstone approve --study <dir> --hours <n> --yes`")
    return 0


def cmd_approve(args) -> int:
    study_dir = Path(args.study)
    resolved = read_json(study_dir / "protocol.resolved.json")
    print(summarize_plan(resolved))
    print(f"\nexpanded jobs: {resolved['expanded_job_count']}")
    hours = resolved["worst_case_budget"]["accelerator_hours"]
    print(f"worst-case allocation: {hours:.3f} accelerator-hours")
    if args.hours is not None:
        resolved["total_accelerator_hours_limit"] = args.hours
        from .util import sha256_json

        resolved.pop("protocol_sha256", None)
        resolved["protocol_sha256"] = sha256_json(resolved)
        write_json(study_dir / "protocol.resolved.json", resolved)
        (study_dir / "protocol.sha256").write_text(resolved["protocol_sha256"] + "\n", encoding="utf-8")
    if not args.yes:
        print("\nnot approved: re-run with --yes after reviewing the plan")
        return 3
    approval = approval_record(resolved, plan_shown=True, limits={"approved_hours": resolved["total_accelerator_hours_limit"]})
    write_json(study_dir / "approval.json", approval)
    print(f"approved at {approval['approved_at']} for {resolved['expanded_job_count']} jobs")
    return 0


def cmd_run(args) -> int:
    study_dir = Path(args.frozen_protocol).parent
    resolved = read_json(args.frozen_protocol)
    approval = read_approval(args.approval)
    if approval["protocol_sha256"] != resolved["protocol_sha256"]:
        _die("approval record does not match the frozen protocol hash")
    gates_path = study_dir / "gates.json"
    if not gates_path.exists():
        _die("no gates.json: run `loadstone validate --stage unit` and `--stage adapter` into the study first")
    gates = read_json(gates_path)
    failing = [g for g in ("unit", "adapter") if not (gates.get(g) or {}).get("passed")]
    if failing:
        _die(f"gates failing: {failing}; no full run while an applicable gate is failing")
    jobs = resolved["job_matrix"]
    if args.jobs:
        keep = {j.strip() for j in args.jobs.split(",")}
        jobs = [j for j in jobs if j["job_id"] in keep]
    limit = _limits_from_file(args.limits)
    limit.eval_per_cell = int(resolved.get("eval_per_cell", limit.eval_per_cell))
    device = resolve_device(resolved.get("environment", {}).get("device"))
    runner = StudyRunner(study_dir, resolved, device=device, limits=limit)
    runner.prepare(approval)
    results = []
    for job in jobs:
        res = runner.run_job(job)
        results.append(res)
        print(f"[{res.status:16s}] {res.job_id}  {res.elapsed_s:7.1f}s", flush=True)
        if res.status in ("CANCELLED", "PAUSED", "BUDGET_EXHAUSTED", "FAILED"):
            break
    build_report(study_dir)
    index_study(study_dir, Path(args.studies_dir) / "loadstone.db")
    return 0


def cmd_pause(args) -> int:
    study_dir = Path(args.study)
    write_json(study_dir / "control.json", {"cancel": args.cancel, "pause": not args.cancel})
    print(f"control request written to {study_dir/'control.json'}")
    return 0


def cmd_resume(args) -> int:
    study_dir = Path(args.study)
    resolved = read_json(study_dir / "protocol.resolved.json")
    approval = read_approval(study_dir / "approval.json")
    jobs = {j["job_id"]: j for j in resolved["job_matrix"]}
    if args.job not in jobs:
        _die(f"unknown job {args.job}; known: {sorted(jobs)}")
    limit = _limits_from_file(args.limits)
    limit.eval_per_cell = int(resolved.get("eval_per_cell", limit.eval_per_cell))
    device = resolve_device(resolved.get("environment", {}).get("device"))
    runner = StudyRunner(study_dir, resolved, device=device, limits=limit)
    runner.prepare(approval)
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(study_dir / "jobs" / args.job)
    res = runner.run_job(jobs[args.job], resume_from=ckpt)
    print(f"[{res.status}] {res.job_id} resumed from {ckpt}")
    build_report(study_dir)
    return 0


def _latest_checkpoint(job_dir: Path) -> Path:
    ckpts = sorted((job_dir / "checkpoints").glob("*.pt"))
    if not ckpts:
        _die(f"no checkpoints in {job_dir}")
    return ckpts[-1]


def cmd_overfit(args) -> int:
    from .gates import run_learning_gate

    resolved = _resolve_for(args.protocol, args.device)
    if args.models:
        keep = {m.strip() for m in args.models.split(",") if m.strip()}
        resolved["model_ids"] = [m for m in resolved["model_ids"] if m in keep]
    device = resolve_device(args.device)
    print(f"learning-plumbing gate: {len(resolved['model_ids'])} models on {device}, "
          f"{args.max_updates} updates / {args.max_minutes} min cap, {args.micro} fixed examples")
    out_path = Path(args.out) if args.out else Path("gates-learning.json")
    doc = run_learning_gate(
        resolved,
        device=device,
        out_path=out_path,
        max_updates=args.max_updates,
        max_seconds=args.max_minutes * 60,
        micro=args.micro,
    )
    for r in doc["rows"]:
        detail = ", ".join(
            f"lr={t['learning_rate']:g}:{t['query_accuracy_on_micro']:.3f}" for t in r["trials"]
        )
        print(f"  {r['model_id']:4s} {detail}  best lr={r['best_learning_rate']:g} "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    return 0 if doc["all_passed"] else 1


def cmd_report(args) -> int:
    print(build_report(Path(args.study), args.format))
    return 0


def cmd_studies(args) -> int:
    db = Path(args.studies_dir) / "loadstone.db"
    rows = list_studies(db)
    if not rows:
        print(f"no studies indexed in {db}")
        return 0
    for r in rows:
        print(f"{r['study_id']:24s} {r['status']:16s} {r['suite']:20s} {r['protocol_sha256'][:16]} {r['path']}")
    return 0


def cmd_manifest(args) -> int:
    doc = load_manifest(args.path)
    ok, problems = verify_manifest(args.path)
    print(json.dumps({"suite": doc["suite"], "version": doc["generator_version"],
                      "heldout_sha256": doc["heldout_sha256"], "verified": ok, "problems": problems}, indent=2))
    return 0 if ok else 1


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loadstone", description=f"Loadstone {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="build and verify dataset manifests")
    dsub = d.add_subparsers(dest="data_cmd", required=True)
    db = dsub.add_parser("build", help="materialise and lock validation/test splits")
    db.add_argument("--out", default="data/mqar-v1")
    db.add_argument("--per-cell", type=int, default=512)
    db.add_argument("--include-ood", action="store_true", help="also materialise length-OOD cells")
    db.add_argument("--generator-seed", type=int, default=500_001)
    db.set_defaults(func=cmd_data_build)

    v = sub.add_parser("validate", help="run gate tests")
    v.add_argument("--protocol", default="protocols/mqar-v1.json")
    v.add_argument("--stage", choices=sorted(GATE_FILES), required=True)
    v.add_argument("--out", default=None, help="write the gate record here")
    v.add_argument("--device", default=None)
    v.set_defaults(func=cmd_validate)

    pl = sub.add_parser("plan", help="resolve a protocol and show the expanded matrix")
    pl.add_argument("--protocol", default="protocols/mqar-v1.json")
    pl.add_argument("--expand-jobs", action="store_true")
    pl.add_argument("--out", default=None)
    pl.add_argument("--device", default=None)
    pl.set_defaults(func=cmd_plan)

    sm = sub.add_parser("smoke", help="run the 20-update plumbing allocation")
    sm.add_argument("--protocol", default="protocols/mqar-v1.json")
    sm.add_argument("--limits", default="protocols/local-limits.json")
    sm.add_argument("--study", default="smoke")
    sm.add_argument("--studies-dir", default="studies")
    sm.add_argument("--models", default=None)
    sm.add_argument("--eval-per-cell", type=int, default=64)
    sm.add_argument("--hours", type=float, default=None)
    sm.add_argument("--device", default=None)
    sm.add_argument("--yes", action="store_true")
    sm.set_defaults(func=cmd_smoke)

    fz = sub.add_parser("freeze", help="freeze a resolved protocol into a study")
    fz.add_argument("--protocol", default="protocols/mqar-v1.json")
    fz.add_argument("--study", required=True)
    fz.add_argument("--studies-dir", default="studies")
    fz.add_argument("--hours", type=float, default=None)
    fz.add_argument("--device", default=None)
    fz.set_defaults(func=cmd_freeze)

    ap = sub.add_parser("approve", help="record a real local approval after reviewing the plan")
    ap.add_argument("--study", required=True)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--yes", action="store_true")
    ap.set_defaults(func=cmd_approve)

    rn = sub.add_parser("run", help="run a frozen, approved, gate-passing study")
    rn.add_argument("--frozen-protocol", required=True)
    rn.add_argument("--approval", required=True)
    rn.add_argument("--jobs", default=None)
    rn.add_argument("--limits", default="protocols/local-limits.json")
    rn.add_argument("--studies-dir", default="studies")
    rn.set_defaults(func=cmd_run)

    pa = sub.add_parser("pause", help="request pause (or cancel) of the active study")
    pa.add_argument("--study", required=True)
    pa.add_argument("--cancel", action="store_true")
    pa.set_defaults(func=cmd_pause)

    rs = sub.add_parser("resume", help="resume a job from a verified checkpoint")
    rs.add_argument("--study", required=True)
    rs.add_argument("--job", required=True)
    rs.add_argument("--checkpoint", default=None)
    rs.add_argument("--limits", default="protocols/local-limits.json")
    rs.set_defaults(func=cmd_resume)

    of = sub.add_parser("overfit", help="Gate 2: bounded overfit on a fixed micro-dataset")
    of.add_argument("--protocol", default="protocols/mqar-v1.json")
    of.add_argument("--models", default=None)
    of.add_argument("--micro", type=int, default=16)
    of.add_argument("--max-updates", type=int, default=500)
    of.add_argument("--max-minutes", type=float, default=5.0)
    of.add_argument("--out", default="gates-learning.json")
    of.add_argument("--device", default=None)
    of.set_defaults(func=cmd_overfit)

    rp = sub.add_parser("report", help="write report.txt, tables and charts")
    rp.add_argument("--study", required=True)
    rp.add_argument("--format", default="txt,csv,json")
    rp.set_defaults(func=cmd_report)

    st = sub.add_parser("studies", help="list indexed studies")
    st.add_argument("--studies-dir", default="studies")
    st.set_defaults(func=cmd_studies)

    mf = sub.add_parser("manifest", help="verify a dataset manifest")
    mf.add_argument("--path", default="data/mqar-v1/dataset-manifest.json")
    mf.set_defaults(func=cmd_manifest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
