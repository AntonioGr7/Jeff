"""The shared prefill must not change any answer. That is the whole contract."""

from __future__ import annotations

import pytest
import torch

from jeff import Jeff, boolean, choice, score
from jeff.engine import _common_prefix_len
from jeff.prompt import Question


STATE = (
    "Order #4417 shipped on 3 March and was marked delivered on 6 March. "
    "The customer wrote in on 9 March saying the box was empty, attached two "
    "photos, and asked for their money back. They have ordered from us eleven "
    "times before and have never opened a ticket."
)

QUESTIONS = [
    boolean("Is the customer asking for a refund?"),
    boolean("Did the customer provide evidence?"),
    boolean("Is this customer likely to be committing fraud?"),
    choice("Which queue should this go to?", ["billing", "shipping", "fraud review"]),
    score("How urgent is this, from 1 to 5?", 1, 5),
]


def test_common_prefix_always_leaves_a_suffix():
    assert _common_prefix_len([[1, 2, 3], [1, 2, 3]]) == 2
    assert _common_prefix_len([[1, 2, 9], [1, 2, 3]]) == 2
    assert _common_prefix_len([[9], [9]]) == 0


def test_question_validation():
    with pytest.raises(ValueError):
        Question("pick one", ("only",))
    with pytest.raises(ValueError):
        Question("pick one", ("a", "a"))
    with pytest.raises(ValueError):
        Question("  ")


@pytest.fixture(scope="module")
def jeff() -> Jeff:
    return Jeff()


@pytest.fixture(scope="module")
def exact() -> Jeff:
    """Float32 on CPU: the only setting where the claim is testable to the bit.

    In bfloat16 the same two paths differ by ~1e-3 in probability, because the
    batch width changes which kernels and reduction orders the backend picks.
    That is precision, not semantics.
    """
    return Jeff(device="cpu", dtype=torch.float32)


def test_shared_prefill_matches_one_at_a_time(exact: Jeff):
    """Batched off a shared cache, vs. each prompt scored alone. Same numbers."""
    together = exact.ask(STATE, QUESTIONS)
    assert exact.stats["prefix_tokens"] > 0
    assert exact.stats["saved_tokens"] > 0

    for shared, question in zip(together, QUESTIONS):
        alone = exact.decide(STATE, question)
        assert shared.choice == alone.choice
        assert shared.probabilities == pytest.approx(alone.probabilities, abs=1e-4)


def test_chunking_matches_one_big_batch(exact: Jeff):
    exact.max_batch = 2
    chunked = exact.ask(STATE, QUESTIONS)
    exact.max_batch = 16
    whole = exact.ask(STATE, QUESTIONS)
    for a, b in zip(chunked, whole):
        assert a.probabilities == pytest.approx(b.probabilities, abs=1e-4)


def test_answers_are_typed_and_total(jeff: Jeff):
    answers = jeff.ask(STATE, QUESTIONS)
    assert len(answers) == len(QUESTIONS)
    for answer, question in zip(answers, QUESTIONS):
        assert answer.choice in question.options
        assert sum(answer.probabilities) == pytest.approx(1.0)
        assert 0.0 <= answer.confidence <= 1.0
    assert 1.0 <= answers[-1].expected_value <= 5.0
    assert jeff.stats["output_tokens"] == 0


def test_temperature_only_moves_confidence(jeff: Jeff):
    sharp = jeff.ask(STATE, QUESTIONS, temperature=0.5)
    flat = jeff.ask(STATE, QUESTIONS, temperature=2.0)
    for a, b in zip(sharp, flat):
        assert a.choice == b.choice
        assert a.confidence >= b.confidence - 1e-9
