"""Vocabulary mass and abstention.

`probabilities` is a softmax over a handful of vocabulary entries out of a
hundred thousand. It cannot tell you whether the model was answering your
question or about to say something else entirely; `mass` can, and these tests
pin the failures it is there to catch.
"""

from __future__ import annotations

import pytest

from jeff import Jeff, boolean, choice
from jeff.prompt import messages

STATE = (
    "Order #4417 was marked delivered on 6 March. The customer wrote in on "
    "9 March saying the box was empty, attached two photos, and asked for "
    "their money back."
)

QUESTIONS = [
    boolean("Is the customer asking for a refund?", id="refund"),
    choice("Which queue?", ["billing", "shipping", "fraud review"], id="queue"),
    choice("What tone should the reply take?", ["apologetic", "neutral", "firm"], id="tone"),
]


@pytest.fixture(scope="module")
def jeff() -> Jeff:
    return Jeff()


def _reencode(jeff: Jeff, **template):
    """Swap in a deliberately wrong prompt build, the way a caller might."""

    def encode(state, question, layout=None, slots=""):
        text = jeff.tokenizer.apply_chat_template(
            messages(state, question, layout, slots), tokenize=False, **template
        )
        return jeff.tokenizer.encode(text, add_special_tokens=False)

    return encode


def test_a_healthy_prompt_captures_the_whole_distribution(jeff: Jeff):
    for answer in jeff.ask(STATE, QUESTIONS):
        assert answer.mass > 0.99
        assert 0.0 <= answer.mass <= 1.0 + 1e-6


@pytest.mark.parametrize(
    "template",
    [
        pytest.param(dict(add_generation_prompt=True, enable_thinking=True), id="thinking-on"),
        pytest.param(dict(add_generation_prompt=False), id="no-generation-prompt"),
    ],
)
def test_mass_collapses_when_the_prompt_is_wrong_but_confidence_does_not(jeff: Jeff, template):
    """The failure this exists for: confident numbers over a menu nobody read.

    With thinking left on, the model is about to emit `<think>`. It still has a
    favourite letter, so the renormalised confidence stays near 1.0 and looks
    entirely healthy. Only the mass shows that none of the real distribution
    was ever on the options.
    """
    healthy = jeff.ask(STATE, QUESTIONS)
    original, jeff._encode, jeff._prefix = jeff._encode, _reencode(jeff, **template), None
    try:
        broken = jeff.ask(STATE, QUESTIONS)
    finally:
        jeff._encode, jeff._prefix = original, None

    for good, bad in zip(healthy, broken):
        assert good.mass > 0.99
        assert bad.mass < 0.01
        assert bad.confidence > 0.5  # still looks decisive, and is worthless


def test_mass_floor_withholds_the_answer(jeff: Jeff):
    original, jeff._encode, jeff._prefix = (
        jeff._encode,
        _reencode(jeff, add_generation_prompt=True, enable_thinking=True),
        None,
    )
    try:
        withheld = jeff.ask(STATE, QUESTIONS, min_mass=0.5)
    finally:
        jeff._encode, jeff._prefix = original, None

    for answer in withheld:
        assert answer.abstained
        assert answer.value is None
        assert answer.choice in answer.question.options  # still says what it would have said
        assert "abstained" in repr(answer)


def test_probability_floor_withholds_only_the_undecided(jeff: Jeff):
    certain = jeff.ask(STATE, QUESTIONS, min_probability=0.9)[0]
    assert certain.abstained is False and certain.value == certain.choice

    for answer in jeff.ask(STATE, QUESTIONS, min_probability=1.0):
        assert answer.abstained and answer.value is None


def test_no_floors_means_no_abstention(jeff: Jeff):
    answers = jeff.ask(STATE, QUESTIONS)
    assert not any(a.abstained for a in answers)
    assert all(a.value == a.choice for a in answers)
    assert jeff.stats["abstained"] == 0


def test_floors_must_be_probabilities(jeff: Jeff):
    for bad in (dict(min_mass=1.5), dict(min_probability=-0.1)):
        with pytest.raises(ValueError, match="probabilities"):
            jeff.ask(STATE, QUESTIONS, **bad)


def test_mass_is_reported_per_layout(jeff: Jeff):
    for answer in jeff.ask(STATE, QUESTIONS, permutations="all"):
        assert len(answer.masses) == len(answer.question.options)
        assert answer.mass == pytest.approx(sum(answer.masses) / len(answer.masses))
