"""The shared prefill must not change any answer. That is the whole contract."""

from __future__ import annotations

import pytest
import torch

from jeff import SLOT_SETS, Jeff, boolean, choice, score
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


def test_permutations_are_off_unless_asked_for(jeff: Jeff):
    plain = jeff.ask(STATE, QUESTIONS)
    assert jeff.stats["forward_rows"] == len(QUESTIONS)
    for answer in plain:
        assert answer.permutations == 1
        assert answer.disagreement == 0.0
    assert plain[0].probabilities == pytest.approx(
        jeff.ask(STATE, QUESTIONS, permutations=1)[0].probabilities
    )


def test_permutations_score_every_layout_and_stay_typed(jeff: Jeff):
    answers = jeff.ask(STATE, QUESTIONS, permutations="all")
    assert jeff.stats["forward_rows"] == sum(len(q.options) for q in QUESTIONS)
    for answer, question in zip(answers, QUESTIONS):
        assert answer.permutations == len(question.options)
        assert answer.choice in question.options
        assert sum(answer.probabilities) == pytest.approx(1.0)
        assert 0.0 <= answer.disagreement <= 1.0
        for sample in answer.samples:
            assert sum(sample) == pytest.approx(1.0)


def test_permutations_are_capped_at_the_option_count(jeff: Jeff):
    answers = jeff.ask(STATE, QUESTIONS, permutations=99)
    for answer, question in zip(answers, QUESTIONS):
        assert answer.permutations == len(question.options)


def test_both_aggregates_produce_a_distribution(jeff: Jeff):
    for aggregate in ("logmean", "mean"):
        for answer in jeff.ask(STATE, QUESTIONS, permutations="all", aggregate=aggregate):
            assert sum(answer.probabilities) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        jeff.ask(STATE, QUESTIONS, aggregate="median")


def test_slot_symbols_are_configurable(jeff: Jeff):
    for slots in ("letters", "digits", "XYZW"):
        for answer in jeff.ask(STATE, QUESTIONS[:4], slots=slots):
            assert sum(answer.probabilities) == pytest.approx(1.0)
            assert set(answer.slots) <= set(SLOT_SETS.get(slots, slots))


def test_a_pool_too_small_for_the_options_is_refused(jeff: Jeff):
    with pytest.raises(ValueError, match="slot symbols"):
        jeff.ask(STATE, [choice("pick", ["a", "b", "c"])], slots="XY")


def test_shuffled_slots_are_per_question_and_reproducible(jeff: Jeff):
    first = jeff.ask(STATE, QUESTIONS, shuffle_slots=True)
    again = jeff.ask(STATE, QUESTIONS, shuffle_slots=True)
    assert [a.slots for a in first] == [a.slots for a in again]
    assert [a.probabilities for a in first] == [a.probabilities for a in again]
    assert [a.slots for a in first] != [a.slots for a in jeff.ask(STATE, QUESTIONS)]
    # Questions of the same width should not all share one favoured symbol.
    pairs = [a.slots for a in first if len(a.slots) == 2]
    assert len(pairs) >= 3 and len(set(pairs)) > 1


def test_shuffling_keeps_one_symbol_set_per_question_so_the_cover_still_cancels(jeff: Jeff):
    for answer in jeff.ask(STATE, QUESTIONS, permutations="all", shuffle_slots=True):
        assert len(answer.slots) == len(answer.question.options)
        assert len(answer.layouts) == len(answer.question.options)
        for option in range(len(answer.question.options)):
            assert sorted(l.index(option) for l in answer.layouts) == list(
                range(len(answer.question.options))
            )


def test_temperature_only_moves_confidence(jeff: Jeff):
    sharp = jeff.ask(STATE, QUESTIONS, temperature=0.5)
    flat = jeff.ask(STATE, QUESTIONS, temperature=2.0)
    for a, b in zip(sharp, flat):
        assert a.choice == b.choice
        assert a.confidence >= b.confidence - 1e-9
