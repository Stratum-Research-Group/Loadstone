"""Dataset manifests: materialise, hash and lock MQAR splits before training."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ..util import sha256_file, sha256_json, write_json
from . import mqar

DEFAULT_GENERATOR_SEED = 500_001
DEFAULT_VALIDATION_SEED = 1001
DEFAULT_TEST_SEED = 2001


def _dump_examples(path: Path, examples: list[mqar.Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "example_id": e.example_id,
            "cell": e.cell,
            "split": e.split,
            "index": e.index,
            "tokens": list(e.tokens),
            "query_positions": list(e.query_positions),
            "targets": list(e.targets),
            "generator_seed": e.generator_seed,
            "content_hash": e.content_hash(),
        }
        for e in examples
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(raw)


def load_examples(path: str | Path) -> list[mqar.Example]:
    with gzip.open(path, "rb") as f:
        payload = json.loads(f.read().decode("utf-8"))
    return [
        mqar.Example(
            split=r["split"],
            cell=r["cell"],
            example_id=r["example_id"],
            tokens=tuple(r["tokens"]),
            query_positions=tuple(r["query_positions"]),
            targets=tuple(r["targets"]),
            generator_seed=r["generator_seed"],
            index=r["index"],
        )
        for r in payload
    ]


def build_manifest(
    outdir: str | Path,
    *,
    per_cell: int = 512,
    generator_seed: int = DEFAULT_GENERATOR_SEED,
    validation_seed: int = DEFAULT_VALIDATION_SEED,
    test_seed: int = DEFAULT_TEST_SEED,
    include_ood: bool = False,
) -> dict:
    """Materialise validation and test sets for every cell, then hash and lock them.

    Validation and test are drawn from independent seeds; test also excludes any
    input already present in validation so the two held-out sets are disjoint.
    """
    outdir = Path(outdir)
    cells = list(mqar.ID_GRID) + (list(mqar.OOD_GRID) if include_ood else [])
    doc: dict = {
        "suite": mqar.SUITE,
        "suite_version": mqar.VERSION,
        "generator_version": f"{mqar.SUITE}@{mqar.VERSION}",
        "generator_seed": generator_seed,
        "alphabet": {
            "size": mqar.ALPHABET_SIZE,
            "pad": mqar.PAD,
            "query": mqar.QUERY,
            "end": mqar.END,
            "keys": [mqar.KEY_BASE, mqar.KEY_BASE + mqar.N_KEYS - 1],
            "values": [mqar.VALUE_BASE, mqar.VALUE_BASE + mqar.N_VALUES - 1],
            "distractors": [mqar.DISTRACTOR_BASE, 255],
        },
        "cells": {mqar.cell_id(L, k): {"L": L, "K": k} for L, k in cells},
        "splits": {"validation": {}, "test": {}},
        "train": {
            "cells": [mqar.cell_id(L, k) for L, k in mqar.ID_GRID],
            "cell_sampling": "uniform_per_example",
            "generation": "deterministic_by_index_with_heldout_rejection",
        },
    }

    validation_hashes: dict[str, set[str]] = {}
    for L, k in cells:
        cid = mqar.cell_id(L, k)
        examples, audit = mqar.build_heldout(
            "validation", length=L, k=k, count=per_cell, generator_seed=validation_seed
        )
        path = outdir / "splits" / "validation" / f"{cid}.jsonl.gz"
        _dump_examples(path, examples)
        doc["splits"]["validation"][cid] = {
            "path": str(path.relative_to(outdir)),
            "sha256": sha256_file(path),
            "count": len(examples),
            "audit": audit,
        }
        validation_hashes[cid] = {e.content_hash() for e in examples}

    for L, k in cells:
        cid = mqar.cell_id(L, k)
        examples, audit = mqar.build_heldout(
            "test",
            length=L,
            k=k,
            count=per_cell,
            generator_seed=test_seed,
            exclude=validation_hashes[cid],
        )
        path = outdir / "splits" / "test" / f"{cid}.jsonl.gz"
        _dump_examples(path, examples)
        doc["splits"]["test"][cid] = {
            "path": str(path.relative_to(outdir)),
            "sha256": sha256_file(path),
            "count": len(examples),
            "audit": audit,
        }

    doc["heldout_sha256"] = sha256_json(
        {s: {c: v["sha256"] for c, v in cells_.items()} for s, cells_ in doc["splits"].items()}
    )
    write_json(outdir / "dataset-manifest.json", doc)
    return doc


def load_manifest(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_manifest(path: str | Path) -> tuple[bool, list[str]]:
    """Re-hash every split file and confirm the manifest has not drifted."""
    path = Path(path)
    doc = load_manifest(path)
    outdir = path.parent
    problems: list[str] = []
    for split, cells in doc["splits"].items():
        for cid, rec in cells.items():
            f = outdir / rec["path"]
            if not f.exists():
                problems.append(f"missing split file {split}/{cid}: {rec['path']}")
                continue
            if sha256_file(f) != rec["sha256"]:
                problems.append(f"split hash mismatch {split}/{cid}")
    expect = sha256_json(
        {s: {c: v["sha256"] for c, v in cells_.items()} for s, cells_ in doc["splits"].items()}
    )
    if expect != doc["heldout_sha256"]:
        problems.append("heldout_sha256 mismatch")
    return not problems, problems


def heldout_hashes(doc: dict) -> set[str]:
    """Canonical input hashes of every locked held-out example."""
    return set(doc.get("_heldout_hashes", ()))
