"""Gate 0: data, solver and scorer unit tests (spec section 12).

Nothing here trains a model. These must pass before any meaningful training.
"""

from __future__ import annotations

import numpy as np
import pytest

from loadstone.datasets import mqar
from loadstone.datasets.manifest import build_manifest, load_examples, verify_manifest
from loadstone.scoring import aggregate, bits_per_byte, cross_entropy_nats, score_item


# ------------------------------------------------------------------ alphabet


def test_alphabet_is_disjoint_and_complete():
    assert mqar.PAD == 0 and mqar.QUERY == 1 and mqar.END == 2
    assert mqar.KEY_BASE == 3
    assert mqar.VALUE_BASE == 67
    assert mqar.DISTRACTOR_BASE == 131
    assert mqar.KEY_BASE + mqar.N_KEYS == mqar.VALUE_BASE
    assert mqar.VALUE_BASE + mqar.N_VALUES == mqar.DISTRACTOR_BASE
    assert mqar.DISTRACTOR_BASE + mqar.N_DISTRACTORS == mqar.ALPHABET_SIZE == 256


# ------------------------------------------------------------------ generator


@pytest.mark.parametrize("length,k", mqar.ID_GRID)
def test_generator_is_deterministic_and_complete(length, k):
    a = mqar.generate_example(length, k, split="train", generator_seed=7, index=3)
    b = mqar.generate_example(length, k, split="train", generator_seed=7, index=3)
    c = mqar.generate_example(length, k, split="train", generator_seed=7, index=4)
    assert a.tokens == b.tokens and a.targets == b.targets
    assert a.tokens != c.tokens
    assert len(a.tokens) == length
    assert len(a.query_positions) == len(a.targets) == k
    assert len(set(a.targets)) == k  # K unique values
    assert a.content_hash() == b.content_hash()


@pytest.mark.parametrize("length,k", mqar.ID_GRID)
def test_layout_and_no_gold_leakage(length, k):
    ex = mqar.generate_example(length, k, split="validation", generator_seed=11, index=5)
    half = length // 2
    query_region = length - 2 * k

    # pairs live strictly in the first half at even starts
    pair_starts = []
    for i in range(half - 1):
        tok = ex.tokens[i]
        if mqar.KEY_BASE <= tok < mqar.KEY_BASE + mqar.N_KEYS:
            assert mqar.VALUE_BASE <= ex.tokens[i + 1] < mqar.VALUE_BASE + mqar.N_VALUES
            assert i % 2 == 0
            pair_starts.append(i)
    assert len(pair_starts) == k
    assert len(set(pair_starts)) == k

    # every filler position in the first half is a distractor
    used = set()
    for s in pair_starts:
        used.update({s, s + 1})
    for i in range(half):
        if i not in used:
            assert mqar.DISTRACTOR_BASE <= ex.tokens[i] <= 255

    # query region is QUERY,key repeated, and no gold value appears at or after
    # the first query position (the answer is never in the input)
    for j in range(k):
        pos = query_region + 2 * j
        assert ex.tokens[pos] == mqar.QUERY
        key = ex.tokens[pos + 1]
        assert mqar.KEY_BASE <= key < mqar.KEY_BASE + mqar.N_KEYS
        assert ex.query_positions[j] == pos + 1
    tail = ex.tokens[query_region:]
    for value in ex.targets:
        assert mqar.VALUE_BASE + value not in tail
    # and the value assigned to a queried key appears nowhere after its query
    pairs = ex.associations()
    for pos in ex.query_positions:
        gold = mqar.VALUE_BASE + pairs[ex.tokens[pos]]
        assert gold not in ex.tokens[pos:]


def test_oracle_solver_is_perfect_and_targets_valid():
    for length, k in mqar.ID_GRID:
        for idx in range(5):
            ex = mqar.generate_example(length, k, split="test", generator_seed=13, index=idx)
            preds = mqar.oracle_predict(ex)
            assert preds == list(ex.targets)
            assert all(0 <= t < mqar.N_VALUES for t in ex.targets)


def test_impossible_cells_fail_loudly():
    with pytest.raises(ValueError):
        mqar.generate_example(64, 32, split="train", generator_seed=1, index=0)
    with pytest.raises(ValueError):
        mqar.generate_example(128, 0, split="train", generator_seed=1, index=0)
    with pytest.raises(ValueError):
        mqar.generate_example(127, 4, split="train", generator_seed=1, index=0)
    with pytest.raises(ValueError):
        mqar.generate_example(128, 4, split="bogus", generator_seed=1, index=0)


def test_ood_lengths_are_supported_by_the_encoding():
    for length, k in mqar.OOD_GRID:
        ex = mqar.generate_example(length, k, split="validation", generator_seed=21, index=0)
        assert ex.length == length
        assert mqar.oracle_predict(ex) == list(ex.targets)


def test_query_distances_are_measured_not_assumed():
    examples, audit = mqar.build_heldout(
        "validation", length=512, k=16, count=4, generator_seed=31
    )
    dist = audit["query_distances"]
    assert 0 < dist["min"] <= dist["p50"] <= dist["max"] < 512


# ------------------------------------------------------------------ held-out


def test_heldout_sets_are_unique_and_disjoint(tmp_path):
    doc = build_manifest(tmp_path, per_cell=32)
    ok, problems = verify_manifest(tmp_path / "dataset-manifest.json")
    assert ok, problems
    base = tmp_path
    for cid in doc["cells"]:
        val = load_examples(base / doc["splits"]["validation"][cid]["path"])
        test = load_examples(base / doc["splits"]["test"][cid]["path"])
        assert len(val) == len(test) == 32
        hv = {e.content_hash() for e in val}
        ht = {e.content_hash() for e in test}
        assert len(hv) == len(val) and len(ht) == len(test)
        assert not (hv & ht), f"{cid}: validation/test overlap"
        for e in val:
            assert e.split == "validation"


def test_training_stream_rejects_heldout_inputs(tmp_path):
    doc = build_manifest(tmp_path, per_cell=8)
    held = set()
    for split in ("validation", "test"):
        for cid, rec in doc["splits"][split].items():
            for e in load_examples(tmp_path / rec["path"]):
                held.add(e.content_hash())
    stream = mqar.TrainingStream(
        cells=list(mqar.ID_GRID), generator_seed=doc["generator_seed"], heldout_hashes=held
    )
    drawn = stream.batch(120)
    assert not ({e.content_hash() for e in drawn} & held)
    assert all(e.split == "train" for e in drawn)
    # resumable: same state reproduces the same stream
    state = stream.state()
    replay = mqar.TrainingStream(
        cells=list(mqar.ID_GRID), generator_seed=doc["generator_seed"], heldout_hashes=held
    )
    replay.load_state(state)
    assert [e.tokens for e in replay.batch(3)] == [e.tokens for e in stream.batch(3)]


def test_manifest_detects_tampering(tmp_path):
    build_manifest(tmp_path, per_cell=8)
    ok, _ = verify_manifest(tmp_path / "dataset-manifest.json")
    assert ok
    victim = next((tmp_path / "splits" / "validation").glob("*.jsonl.gz"))
    data = bytearray(victim.read_bytes())
    data[0] ^= 0xFF
    victim.write_bytes(bytes(data))
    ok, problems = verify_manifest(tmp_path / "dataset-manifest.json")
    assert not ok and problems
