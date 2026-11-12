"""Gate 0: scorer unit tests (spec section 12).

Scorers are pure functions. A uniform distribution over 256 bytes is exactly
8 bits/byte; hand-computed examples pin the nats-to-bits conversion; failures
stay in the planned denominator.
"""

from __future__ import annotations

import numpy as np
import pytest

from loadstone.datasets import mqar
from loadstone.scoring import aggregate, bits_per_byte, cross_entropy_nats, score_item





def test_gold_predictions_score_full_marks():
    ex = mqar.generate_example(128, 4, split="test", generator_seed=3, index=0)
    logits = np.full((ex.k, 64), -20.0)
    for i, t in enumerate(ex.targets):
        logits[i, t] = 20.0
    res = score_item(ex, logits)
    assert res.correct == res.n_queries
    assert res.sequence_correct
    assert res.nll_sum < 1e-6


def test_wrong_predictions_score_zero_exact_accuracy():
    ex = mqar.generate_example(128, 4, split="test", generator_seed=3, index=1)
    logits = np.full((ex.k, 64), -20.0)
    for i, t in enumerate(ex.targets):
        logits[i, (t + 1) % 64] = 20.0
    res = score_item(ex, logits)
    assert res.correct == 0 and not res.sequence_correct


def test_aggregate_keeps_failures_in_the_denominator():
    ex = mqar.generate_example(128, 4, split="test", generator_seed=3, index=2)
    good = np.full((ex.k, 64), -20.0)
    for i, t in enumerate(ex.targets):
        good[i, t] = 20.0
    items = [score_item(ex, good), score_item(ex, None, failure="timeout")]
    agg = aggregate(items, planned=4)
    assert agg["query_accuracy"] == 0.5
    assert agg["coverage"] == 0.25  # 1 of 4 planned examples produced a score
    assert agg["cells"]["L128_K4"]["failed_examples"] == 1


def test_cross_entropy_and_bits_per_byte_are_exact():
    # uniform over 256 bytes == exactly 8 bits per byte
    nll = cross_entropy_nats(np.zeros((1, 256)), np.array([7]))
    assert bits_per_byte(float(nll.sum()), 1) == pytest.approx(8.0, abs=1e-9)
    # hand-computed two-class example: logits [0, ln 3] -> p=(0.25, 0.75)
    nll = cross_entropy_nats(np.array([[0.0, float(np.log(3.0))]]), np.array([0]))
    assert float(nll[0]) == pytest.approx(-np.log(0.25))
    assert bits_per_byte(float(nll[0]), 1) == pytest.approx(-np.log2(0.25))


def test_scorer_rejects_malformed_shapes():
    ex = mqar.generate_example(128, 4, split="test", generator_seed=3, index=3)
    with pytest.raises(ValueError):
        score_item(ex, np.zeros((3, 64)))
    with pytest.raises(ValueError):
        cross_entropy_nats(np.zeros((2, 4)), np.array([4, 0]))
