"""Suite A: controlled MQAR (`mqar_controlled_v1`).

Task (spec section 7), implemented exactly:

Symbol alphabet, 256 symbols::

    PAD        = 0
    QUERY      = 1
    END        = 2
    keys       = 3   .. 66    (64 keys)
    values     = 67  .. 130   (64 values)
    distractors= 131 .. 255   (125 filler symbols)

An example of length L:

* K unique keys paired with K unique values.
* Those K pairs occupy 2K distinct *even* start positions inside the first half
  of the sequence (key at s, value at s+1). Requires 2K <= L/2.
* Every remaining first-half position is an independently sampled distractor.
* The final 2K positions hold K query slots: QUERY then the queried key. The
  query key is the scored position; the gold value never appears in the input.
* Causal mask; loss is cross entropy over the 64 value classes at query-key
  positions only. No prefix-value supervision in this variant.

The generator is a pure function of (generator_seed, split, cell, index) so any
example can be regenerated from its identity alone. Independent streams are
derived per split, so validation/test draws never depend on training.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..util import rng_for

SUITE = "mqar_controlled_v1"
VERSION = "1.0.0"

PAD = 0
QUERY = 1
END = 2
KEY_BASE = 3
N_KEYS = 64
VALUE_BASE = KEY_BASE + N_KEYS  # 67
N_VALUES = 64
DISTRACTOR_BASE = VALUE_BASE + N_VALUES  # 131
N_DISTRACTORS = 256 - DISTRACTOR_BASE  # 125
ALPHABET_SIZE = 256
CHANCE_ACCURACY = 1.0 / N_VALUES

#: in-distribution starter grid (spec 7 "Starter grid"): nine cells.
ID_GRID: tuple[tuple[int, int], ...] = tuple(
    (length, k) for length in (128, 256, 512) for k in (4, 8, 16)
)
#: length-OOD cells; only generated after the encoding gate confirms support.
OOD_GRID: tuple[tuple[int, int], ...] = tuple(
    (length, k) for length in (1024, 2048) for k in (4, 8, 16)
)


def cell_id(length: int, k: int) -> str:
    return f"L{length}_K{k}"


def parse_cell(cid: str) -> tuple[int, int]:
    left, right = cid.split("_")
    return int(left[1:]), int(right[1:])


@dataclass(frozen=True)
class Example:
    """One canonical example plus its private scoring targets."""

    split: str
    cell: str
    example_id: str
    tokens: tuple[int, ...]
    query_positions: tuple[int, ...]
    targets: tuple[int, ...]  # value class indices 0..63
    generator_seed: int
    index: int

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def k(self) -> int:
        return len(self.targets)

    def content_hash(self) -> str:
        arr = np.asarray(self.tokens, dtype=np.uint8)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def associations(self) -> dict[int, int]:
        """Oracle reference map: key id -> value class index, read from the input.

        This is the independent solver: it re-reads the emitted symbols rather
        than trusting generator bookkeeping, so a placement bug shows up as a
        solver disagreement instead of silently passing.
        """
        pairs: dict[int, int] = {}
        half = self.length // 2
        i = 0
        while i < half - 1:
            tok = self.tokens[i]
            if KEY_BASE <= tok < KEY_BASE + N_KEYS:
                nxt = self.tokens[i + 1]
                if VALUE_BASE <= nxt < VALUE_BASE + N_VALUES:
                    pairs[tok] = nxt - VALUE_BASE
                    i += 2
                    continue
            i += 1
        return pairs


def validate_cell(length: int, k: int) -> None:
    if length < 4 or length % 2 != 0:
        raise ValueError(f"L must be even and >= 4, got {length}")
    if not (1 <= k <= N_VALUES):
        raise ValueError(f"K must be in 1..{N_VALUES}, got {k}")
    if 2 * k > length // 2:
        raise ValueError(f"cell L={length} K={k} is impossible: requires 2K <= L/2")
    if 2 * k > length:
        raise ValueError(f"cell L={length} K={k}: query region exceeds sequence")
    if length // 4 < k:
        raise ValueError(f"cell L={length} K={k}: not enough even pair slots in first half")


def generate_example(length: int, k: int, *, split: str, generator_seed: int, index: int) -> Example:
    """Deterministic example for (split, generator_seed, cell, index)."""
    validate_cell(length, k)
    split = _normalise_split(split)
    rng = rng_for("mqar", VERSION, generator_seed, split, length, k, index)

    keys = rng.choice(N_KEYS, size=k, replace=False)
    values = rng.choice(N_VALUES, size=k, replace=False)
    slots = np.arange(0, length // 2, 2)  # even start positions in the first half
    starts = np.sort(rng.choice(slots, size=k, replace=False))

    tokens = (DISTRACTOR_BASE + rng.integers(0, N_DISTRACTORS, size=length)).astype(np.int64)
    for j in range(k):
        s = int(starts[j])
        tokens[s] = KEY_BASE + int(keys[j])
        tokens[s + 1] = VALUE_BASE + int(values[j])

    q_base = length - 2 * k
    query_positions: list[int] = []
    for j in range(k):
        pos = q_base + 2 * j
        tokens[pos] = QUERY
        tokens[pos + 1] = KEY_BASE + int(keys[j])
        query_positions.append(pos + 1)

    return Example(
        split=split,
        cell=cell_id(length, k),
        example_id=f"{SUITE}:{split}:{cell_id(length, k)}:{index}",
        tokens=tuple(int(t) for t in tokens),
        query_positions=tuple(query_positions),
        targets=tuple(int(v) for v in values),
        generator_seed=generator_seed,
        index=index,
    )


def _normalise_split(split: str) -> str:
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split {split!r}")
    return split


# --------------------------------------------------------------- oracle solver


def oracle_predict(example: Example) -> list[int]:
    """Reference solver: 100% accuracy by construction, independent of the model."""
    pairs = example.associations()
    out = []
    for pos in example.query_positions:
        key = example.tokens[pos]
        if key not in pairs:
            raise KeyError(f"{example.example_id}: query key {key} has no association")
        out.append(pairs[key])
    return out


# --------------------------------------------------------------- held-out sets


def build_heldout(
    split: str,
    *,
    length: int,
    k: int,
    count: int,
    generator_seed: int,
    max_attempts_factor: int = 40,
    exclude: set[str] | None = None,
) -> tuple[list[Example], dict]:
    """Materialise `count` unique held-out examples for one cell and split.

    Duplicates are rejected by canonical content hash (also against `exclude`,
    which carries the other held-out split). If the finite space cannot supply
    `count` unique inputs, the returned set is deterministically smaller and
    the audit records the shortfall rather than inventing examples.
    """
    split = _normalise_split(split)
    seen: set[str] = set(exclude or ())
    out: list[Example] = []
    attempts = 0
    duplicates = 0
    limit = count * max_attempts_factor
    index = 0
    while len(out) < count and attempts < limit:
        ex = generate_example(length, k, split=split, generator_seed=generator_seed, index=index)
        attempts += 1
        index += 1
        h = ex.content_hash()
        if h in seen:
            duplicates += 1
            continue
        seen.add(h)
        out.append(ex)
    audit = {
        "requested": count,
        "delivered": len(out),
        "attempts": attempts,
        "duplicates_rejected": duplicates,
        "shortfall": max(0, count - len(out)),
        "unique_inputs": len({e.content_hash() for e in out}),
        "query_distances": _distance_stats(out),
    }
    return out, audit


def _distance_stats(examples: list[Example]) -> dict:
    dists = [pos - s for e in examples for pos, s in _pair_positions(e)]
    if not dists:
        return {}
    arr = np.asarray(dists, dtype=np.int64)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _pair_positions(example: Example) -> Iterator[tuple[int, int]]:
    """(query position, matched key position) for every query in the example."""
    pairs = example.associations()
    positions: dict[int, int] = {}
    half = example.length // 2
    for i in range(half - 1):
        tok = example.tokens[i]
        if KEY_BASE <= tok < KEY_BASE + N_KEYS and VALUE_BASE <= example.tokens[i + 1] < VALUE_BASE + N_VALUES:
            positions.setdefault(tok, i)
    for pos in example.query_positions:
        key = example.tokens[pos]
        if key in pairs and key in positions:
            yield pos, positions[key]


# --------------------------------------------------------------- training stream


class TrainingStream:
    """Deterministic, resumable sampler over the training distribution.

    Cells are sampled uniformly; examples are generated by a monotonically
    increasing counter so the stream is reproducible from its RNG state and
    cursor. Any example whose canonical input hash is in the held-out set is
    skipped (deterministically) and counted.
    """

    def __init__(
        self,
        *,
        cells: list[tuple[int, int]],
        generator_seed: int,
        heldout_hashes: set[str],
        max_resample: int = 32,
    ) -> None:
        self.cells = list(cells)
        self.generator_seed = generator_seed
        self.heldout_hashes = heldout_hashes
        self.max_resample = max_resample
        self.rng = rng_for("mqar-train-stream", VERSION, generator_seed)
        self.counter = 0
        self.skipped = 0

    def draw(self) -> list[Example]:
        out: list[Example] = []
        for _ in range(self.max_resample):
            ci = int(self.rng.integers(0, len(self.cells)))
            length, k = self.cells[ci]
            ex = generate_example(
                length, k, split="train", generator_seed=self.generator_seed, index=self.counter
            )
            self.counter += 1
            if ex.content_hash() in self.heldout_hashes:
                self.skipped += 1
                continue
            out.append(ex)
            break
        if not out:
            raise RuntimeError("training stream could not draw a non-held-out example")
        return out

    def batch(self, size: int) -> list[Example]:
        return [self.draw()[0] for _ in range(size)]

    def state(self) -> dict:
        return {
            "counter": self.counter,
            "skipped": self.skipped,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state(self, state: dict) -> None:
        self.counter = int(state["counter"])
        self.skipped = int(state["skipped"])
        self.rng.bit_generator.state = state["rng_state"]
