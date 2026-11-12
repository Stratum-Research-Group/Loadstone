"""Gate 2: learning-plumbing gates on a fixed micro-dataset.

These are *plumbing* checks, not held-out capability evidence. A model that fails
them may have a training-setting bug rather than an architectural limit, so the
failure is kept visible instead of being explained away.

The gate sweeps the protocol's screening learning rates, because a bounded
overfit check at a single rate mostly measures how that rate happens to suit the
branch structure. Passing at any screening rate means the learning path works;
the rate that passed is recorded.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .datasets import mqar
from .models import encode_batch, gather_query_logits, registered_adapters
from .protocol import load_protocol, resolve
from .util import write_json


def fixed_micro_dataset(count: int = 16, *, length: int = 128, k: int = 4, seed: int = 900_001):
    """A fixed, tiny, same-distribution training set (regenerated deterministically)."""
    return [
        mqar.generate_example(length, k, split="train", generator_seed=seed, index=i)
        for i in range(count)
    ]


def _build(model_id: str, resolved: dict, device: str, seed: int):
    size = resolved["capability_matrix"][model_id]["size"]
    config = {
        "width": size["width"],
        "depth": size["depth"],
        "n_heads": size["heads"],
        "target_trainable_parameters": size["target"],
        "parameter_tolerance_fraction": size["tolerance"],
        "parameter_matched_claimed": size["matched"],
        "parameter_deviation": size["deviation"],
    }
    adapter = registered_adapters(
        source_commit=resolved["adapter_commits"].get(model_id, "unknown"), device=device
    )[model_id]
    adapter.build(config, initialization_seed=seed)
    return adapter


def overfit_trial(
    model_id: str,
    resolved: dict,
    *,
    device: str,
    lr: float,
    max_updates: int,
    max_seconds: float,
    micro: int,
    seed: int,
) -> dict:
    adapter = _build(model_id, resolved, device, seed)
    examples = fixed_micro_dataset(micro)
    batch = encode_batch(examples, device)
    optimizer = torch.optim.AdamW(
        adapter.model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    t0 = time.perf_counter()
    losses = []
    step = 0
    while step < max_updates and time.perf_counter() - t0 < max_seconds:
        out = adapter.train_step(batch, optimizer)
        losses.append(out["loss"])
        step += 1
    adapter.model.eval()
    with torch.no_grad():
        logits = adapter.model(batch.tokens, batch.pad_mask)
        q = gather_query_logits(logits, batch.query_positions)
        correct = int(((q.argmax(-1) == batch.targets) & (batch.targets >= 0)).sum())
        total = int((batch.targets >= 0).sum())
    return {
        "learning_rate": lr,
        "updates": step,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "query_accuracy_on_micro": correct / total,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "passed": (correct / total) >= 0.95,
    }


def overfit_gate(
    model_id: str,
    resolved: dict,
    *,
    device: str,
    max_updates: int = 500,
    max_seconds: float = 300.0,
    micro: int = 16,
    seed: int = 17,
    lrs: list[float] | None = None,
) -> dict:
    lrs = lrs or list(resolved.get("learning_rates") or [1e-3])
    trials = [
        overfit_trial(
            model_id,
            resolved,
            device=device,
            lr=lr,
            max_updates=max_updates,
            max_seconds=max_seconds,
            micro=micro,
            seed=seed,
        )
        for lr in lrs
    ]
    best = max(trials, key=lambda t: t["query_accuracy_on_micro"])
    return {
        "gate": "learning_plumbing",
        "model_id": model_id,
        "device": device,
        "micro_examples": micro,
        "trials": trials,
        "best_learning_rate": best["learning_rate"],
        "best_accuracy_on_micro": best["query_accuracy_on_micro"],
        "passed": any(t["passed"] for t in trials),
        "note": "Same-dataset overfit check. It proves the learning path works; it is not "
                "held-out capability evidence and must never be reported as a score. Passing "
                "at any screening learning rate counts; the rate is recorded.",
    }


def run_learning_gate(resolved: dict, *, device: str, out_path: Path, **kwargs) -> dict:
    rows = [overfit_gate(mid, resolved, device=device, **kwargs) for mid in resolved["model_ids"]]
    doc = {
        "gate": "learning_plumbing",
        "device": device,
        "all_passed": all(r["passed"] for r in rows),
        "rows": rows,
    }
    write_json(out_path, doc)
    return doc


def resolve_for_gate(protocol_path: str, device: str | None = None) -> dict:
    return resolve(load_protocol(protocol_path), base_dir=Path.cwd(), device=device)
