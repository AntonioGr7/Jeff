"""Slot symbols: choosing them, drawing them, and measuring what they cost."""

from __future__ import annotations

import math

import pytest

from jeff.prompt import Answer, Question, cyclic_layouts
from jeff.slots import (
    SLOT_SETS,
    estimate_slot_bias,
    resolve_slots,
    spread,
    take_slots,
)


def test_presets_and_literals():
    assert resolve_slots("letters") == SLOT_SETS["letters"]
    assert resolve_slots("XYZ") == "XYZ"


def test_rejects_unusable_pools():
    with pytest.raises(ValueError):
        resolve_slots("A")
    with pytest.raises(ValueError):
        resolve_slots("AAB")


def test_unkeyed_draw_is_the_conventional_one():
    assert take_slots("ABCDEF", 3) == "ABC"


def test_keyed_draw_is_a_reproducible_subset():
    first = take_slots(SLOT_SETS["letters"], 4, key="urgency")
    assert first == take_slots(SLOT_SETS["letters"], 4, key="urgency")
    assert len(first) == len(set(first)) == 4
    assert set(first) <= set(SLOT_SETS["letters"])


def test_the_draw_moves_with_question_and_seed():
    pool = SLOT_SETS["extended"]
    draws = {take_slots(pool, 4, key=str(k)) for k in range(12)}
    assert len(draws) > 6  # different questions get different symbols
    assert take_slots(pool, 4, key="q", seed=1) != take_slots(pool, 4, key="q", seed=2)


def test_too_many_options_for_the_pool_says_so():
    with pytest.raises(ValueError, match="digits only go up to ten"):
        take_slots(SLOT_SETS["digits"], 11)
    with pytest.raises(ValueError, match="slot symbols"):
        take_slots("ABC", 4)


def _biased_answer(scores, bias, slots):
    """An Answer as a model with exactly this symbol bias would have produced."""
    question = Question("probe", tuple(f"option {i}" for i in range(len(scores))))
    samples, layouts = [], []
    for layout in cyclic_layouts(len(scores), "all"):
        shown = [scores[option] + bias[slots[j]] for j, option in enumerate(layout)]
        canonical = [0.0] * len(scores)
        for j, option in enumerate(layout):
            canonical[option] = shown[j]
        total = sum(math.exp(v) for v in canonical)
        samples.append(tuple(math.exp(v) / total for v in canonical))
        layouts.append(layout)
    return Answer(question, samples[0], (), tuple(samples), tuple(layouts), slots)


def test_estimator_recovers_an_injected_symbol_bias():
    bias = {"A": 1.5, "B": 0.0, "C": -0.5, "D": -1.0}
    centre = sum(bias.values()) / len(bias)
    answers = [
        _biased_answer([0.0, 0.4, 1.1, 2.0], bias, "ABCD"),
        _biased_answer([1.0, 0.0, 0.3, 0.2], bias, "ABCD"),
    ]
    estimated = estimate_slot_bias(answers)
    for symbol, value in bias.items():
        assert estimated[symbol] == pytest.approx(value - centre, abs=1e-6)


def test_estimator_reports_no_bias_when_there_is_none():
    flat = dict.fromkeys("ABCD", 0.0)
    estimated = estimate_slot_bias([_biased_answer([0.0, 1.0, 2.0, 3.0], flat, "ABCD")])
    assert spread(estimated) == pytest.approx(0.0, abs=1e-6)


def test_estimator_needs_a_full_cover():
    question = Question("probe", ("a", "b"))
    with pytest.raises(ValueError, match="permutations='all'"):
        estimate_slot_bias([Answer(question, (0.5, 0.5), (0.0, 0.0))])
