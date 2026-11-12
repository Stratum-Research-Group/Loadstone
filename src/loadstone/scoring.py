"""Scoring: pure functions, reference solvers and scorer fixtures.

Scorers never mutate, never read files and never look at model internals beyond
the declared prediction surface (`logits` at query positions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SCORER_VERSION = "mqar-scorer@1.0.0"


@dataclass
class ItemResult:
    example_id: str
    cell: str
    n_queries: int
    correct: int
    nll_sum: float  # nats, summed over queries
    predicted: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    failure: str | None = None

    @property
    def sequence_correct(self) -> bool:
        return self.failure is None and self.correct == self.n_queries

    @property
    def query_accuracy(self) -> float:
        if self.failure is not None or self.n_queries == 0:
            return 0.0
        return self.correct / self.n_queries


def cross_entropy_nats(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-row cross entropy in nats. `logits` [P, C] as float64."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got shape {logits.shape}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits/targets length mismatch")
    if targets.size and (targets.min() < 0 or targets.max() >= logits.shape[1]):
        raise ValueError("target out of range")
    shifted = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    logp = shifted - logsumexp
    return -logp[np.arange(logits.shape[0]), targets]


def score_item(
    example, logits: np.ndarray | None, *, failure: str | None = None
) -> ItemResult:
    """Score one example. `logits` is [n_queries, 64] aligned with query order."""
    targets = np.asarray(example.targets, dtype=np.int64)
    if failure is not None:
        return ItemResult(
            example_id=example.example_id,
            cell=example.cell,
            n_queries=len(targets),
            correct=0,
            nll_sum=float("nan"),
            failure=failure,
        )
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(targets), 64):
        raise ValueError(f"{example.example_id}: expected logits {(len(targets), 64)}, got {logits.shape}")
    nll = cross_entropy_nats(logits, targets)
    predicted = logits.argmax(axis=1)
    return ItemResult(
        example_id=example.example_id,
        cell=example.cell,
        n_queries=len(targets),
        correct=int((predicted == targets).sum()),
        nll_sum=float(nll.sum()),
        predicted=predicted.tolist(),
        targets=targets.tolist(),
    )


def aggregate(items: list[ItemResult], *, planned: int | None = None) -> dict:
    """Aggregate per-example results. Macro averages are over cells.

    Unanswered / failed items stay in the denominator as zero-accuracy failures.
    """
    if not items:
        raise ValueError("no items to aggregate")
    by_cell: dict[str, list[ItemResult]] = {}
    for it in items:
        by_cell.setdefault(it.cell, []).append(it)

    cell_rows = {}
    for cell, rows in sorted(by_cell.items()):
        queries = sum(r.n_queries for r in rows)
        correct = sum(r.correct for r in rows)
        nll = sum(r.nll_sum for r in rows if r.failure is None)
        scored_queries = sum(r.n_queries for r in rows if r.failure is None)
        failures = sum(1 for r in rows if r.failure is not None)
        cell_rows[cell] = {
            "examples": len(rows),
            "failed_examples": failures,
            "queries": queries,
            "query_accuracy": (correct / queries) if queries else 0.0,
            "sequence_accuracy": (sum(r.sequence_correct for r in rows) / len(rows)) if rows else 0.0,
            "cross_entropy_nats": (nll / scored_queries) if scored_queries else float("nan"),
            "coverage": ((len(rows) - failures) / len(rows)) if rows else 0.0,
        }

    total_queries = sum(r.n_queries for r in items)
    total_correct = sum(r.correct for r in items)
    return {
        "examples": len(items),
        "planned_examples": planned if planned is not None else len(items),
        "coverage": (len([r for r in items if r.failure is None]) / (planned or len(items))),
        "query_accuracy": (total_correct / total_queries) if total_queries else 0.0,
        "cell_macro_query_accuracy": float(
            np.mean([c["query_accuracy"] for c in cell_rows.values()])
        ),
        "cell_macro_sequence_accuracy": float(
            np.mean([c["sequence_accuracy"] for c in cell_rows.values()])
        ),
        "cells": cell_rows,
        "scorer_version": SCORER_VERSION,
    }


def bits_per_byte(nll_nats: float, n_bytes: int) -> float:
    """Bits per byte from summed causal NLL (spec section 9)."""
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return nll_nats / (n_bytes * math.log(2))
