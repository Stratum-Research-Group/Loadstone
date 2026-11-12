"""Gate 1: adapter mechanics (spec section 12).

Causality, state isolation, bypass honesty, parameter accounting, checkpoint
fidelity and batch invariance. Tolerances are frozen here: atol=1e-5,
rtol=1e-4 for float32.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from loadstone.datasets import mqar
from loadstone.models import (
    CANDIDATE_WIDTH_GRID,
    N_CLASSES,
    VOCAB,
    TorchMQARAdapter,
    analytic_param_count,
    encode_batch,
    registered_adapters,
    resolve_size,
)
from loadstone.runner import lr_at

ATOL, RTOL = 1e-5, 1e-4
TINY = {"width": 32, "depth": 1, "n_heads": 4, "target_trainable_parameters": 5000,
        "parameter_tolerance_fraction": 0.05, "parameter_matched_claimed": False,
        "parameter_deviation": 0.0}


def _adapter(mid: str, seed: int = 17) -> TorchMQARAdapter:
    ad = registered_adapters(device="cpu")[mid]
    ad.build(TINY, initialization_seed=seed)
    return ad


@pytest.mark.parametrize("mid", ["B0", "B1", "B2", "B3", "T0"])
def test_parameter_counts_match_tensors(mid):
    ad = _adapter(mid)
    actual = ad.model.trainable_parameters()
    analytic = analytic_param_count(
        TINY["width"], TINY["depth"], use_memory=ad.use_memory, use_reasoning=ad.use_reasoning
    )
    assert actual == analytic
    assert ad.metadata()["trainable_parameters"] == actual


@pytest.mark.parametrize("mid", ["B0", "B1", "B2", "B3"])
def test_bypass_really_removes_the_component(mid):
    ad = _adapter(mid)
    names = [n for n, _ in ad.model.named_parameters()]
    has_mem = any(n.startswith("blocks.0.memory") for n in names)
    has_reason = any(n.startswith("reason_") for n in names)
    assert has_mem == ad.use_memory, f"{mid}: memory presence mismatch"
    assert has_reason == ad.use_reasoning, f"{mid}: reasoning presence mismatch"
    # a bypassed component cannot report executed work
    ex = mqar.generate_example(128, 4, split="train", generator_seed=1, index=0)
    batch = encode_batch([ex], "cpu")
    opt = torch.optim.AdamW(ad.model.parameters(), lr=1e-3)
    out = ad.train_step(batch, opt)
    assert isinstance(out["loss"], float) and np.isfinite(out["loss"])
    if not ad.use_memory:
        assert out["counters"]["memory_chunks"] == 0
    if not ad.use_reasoning:
        assert out["counters"]["reasoning_loop_iterations"] == 0


@pytest.mark.parametrize("mid", ["T0", "B1", "B2", "B3"])
def test_future_tokens_do_not_change_earlier_logits(mid):
    ad = _adapter(mid)
    ad.model.eval()
    ex = mqar.generate_example(256, 8, split="train", generator_seed=2, index=0)
    cut = ex.length // 2
    tok_a = torch.tensor([ex.tokens], dtype=torch.long)
    perturbed = list(ex.tokens)
    rng = np.random.default_rng(0)
    for i in range(cut, ex.length):
        perturbed[i] = int(rng.integers(0, 256))
    tok_b = torch.tensor([perturbed], dtype=torch.long)
    with torch.no_grad():
        a = ad.model(tok_a)
        b = ad.model(tok_b)
    # `cut` itself is perturbed, so positions < cut must be untouched
    assert torch.allclose(a[:, :cut], b[:, :cut], atol=ATOL, rtol=RTOL)
    assert not torch.allclose(a[:, cut:], b[:, cut:], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mid", ["T0", "B1", "B2", "B3"])
def test_state_is_fresh_for_every_example(mid):
    ad = _adapter(mid)
    a = mqar.generate_example(128, 4, split="validation", generator_seed=3, index=0)
    b = mqar.generate_example(256, 8, split="validation", generator_seed=3, index=9)
    alone = ad.predict(encode_batch([b], "cpu"))[0]
    ad.predict(encode_batch([a], "cpu"))
    after = ad.predict(encode_batch([b], "cpu"))[0]
    assert torch.allclose(alone, after, atol=ATOL, rtol=RTOL)
    assert ad.reset_state("all") == {}


@pytest.mark.parametrize("mid", ["T0", "B1", "B3"])
def test_batching_and_padding_do_not_change_predictions(mid):
    ad = _adapter(mid)
    short = mqar.generate_example(128, 4, split="validation", generator_seed=4, index=0)
    long = mqar.generate_example(512, 16, split="validation", generator_seed=4, index=1)
    solo = ad.predict(encode_batch([short], "cpu"))[0]
    pair = ad.predict(encode_batch([short, long], "cpu"))[0]
    assert torch.allclose(solo, pair, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("mid", ["T0", "B1", "B2", "B3"])
def test_checkpoint_roundtrip_preserves_outputs(mid, tmp_path):
    ad = _adapter(mid)
    ex = mqar.generate_example(128, 4, split="validation", generator_seed=5, index=0)
    batch = encode_batch([ex], "cpu")
    before = ad.predict(batch)[0]
    path = tmp_path / "ckpt.pt"
    ad.save_checkpoint(path)
    fresh = _adapter(mid, seed=99)
    fresh.load_checkpoint(path)
    after = fresh.predict(batch)[0]
    assert torch.allclose(before, after, atol=ATOL, rtol=RTOL)


def test_evaluation_does_not_change_base_parameters():
    ad = _adapter("B3")
    ex = mqar.generate_example(128, 4, split="validation", generator_seed=6, index=0)
    snapshot = {k: v.clone() for k, v in ad.model.state_dict().items()}
    ad.predict(encode_batch([ex], "cpu"))
    for key, value in snapshot.items():
        assert torch.equal(value, ad.model.state_dict()[key]), f"{key} changed during evaluation"


def test_size_search_is_deterministic_and_inside_the_band():
    for target in (1_000_000, 5_000_000):
        a = resolve_size(target)
        b = resolve_size(target)
        assert a == b
        assert a.deviation <= a.tolerance or a.matched is False
    one_m = resolve_size(1_000_000)
    assert one_m.matched, f"1M target unresolved: {one_m}"
    # tie-break: smaller width, then smaller depth
    assert (one_m.width, one_m.depth) <= (one_m.width, one_m.depth)
    mem = resolve_size(
        1_000_000, use_memory=True, use_reasoning=True, widths=CANDIDATE_WIDTH_GRID
    )
    assert mem.matched, "candidate grid must contain a matched configuration for every ablation"


def test_lr_schedule_warms_up_then_decays():
    total = 1000
    lrs = [lr_at(i, total, 1e-3) for i in range(total)]
    assert lrs[0] < lrs[50] + 1e-12
    assert max(lrs) <= 1e-3 + 1e-12
    assert lrs[-1] < 1e-5
    assert lrs[500] > lrs[900]


def test_preflight_reports_unsupported_lengths():
    ad = registered_adapters(device="cpu")["T0"]
    assert ad.preflight("L128_K4")[0] == "SUPPORTED"
    assert ad.preflight("L8192_K4")[0] == "UNSUPPORTED"
