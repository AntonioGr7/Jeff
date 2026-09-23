"""Batch sizing and option rubrics — both testable without loading a model."""

from __future__ import annotations

import pytest

from jeff import Question, choice
from jeff.engine import Jeff
from jeff.prompt import messages


class Sizer:
    """Just the two dials `_chunks` reads, so the packing is testable on its own."""

    def __init__(self, max_batch: int, max_batch_tokens: int) -> None:
        self.max_batch, self.max_batch_tokens = max_batch, max_batch_tokens

    chunks = Jeff._chunks


def prompts(prefix: int, *suffixes: int) -> list[list[int]]:
    return [[0] * (prefix + length) for length in suffixes]


def test_short_prefix_fills_the_batch():
    rows = prompts(50, *([10] * 40))
    sizes = [len(c) for c in Sizer(16, 16384).chunks(rows, 50)]
    assert sizes == [16, 16, 8]


def test_a_long_prefix_narrows_the_batch_instead_of_failing():
    """Every row carries its own copy of the prefix, so width has to give."""
    rows = prompts(9000, *([40] * 6))
    sizes = [len(c) for c in Sizer(16, 16384).chunks(rows, 9000)]
    assert sizes == [1] * 6


def test_the_budget_is_respected_whatever_the_shape():
    for prefix, count, width, budget in ((100, 30, 16, 4096), (2000, 12, 8, 8192), (500, 7, 4, 1200)):
        rows = prompts(prefix, *[(i % 5) * 10 + 5 for i in range(count)])
        chunks = list(Sizer(width, budget).chunks(rows, prefix))
        assert sum(len(c) for c in chunks) == count
        for chunk in chunks:
            longest = max(len(p) - prefix for p in chunk)
            assert len(chunk) <= width
            assert len(chunk) == 1 or len(chunk) * (prefix + longest) <= budget


def test_every_row_is_scored_exactly_once_and_in_order():
    rows = prompts(80, *range(1, 26))
    assert [p for chunk in Sizer(7, 4096).chunks(rows, 80) for p in chunk] == rows


def test_a_single_row_is_never_dropped_for_being_too_big():
    rows = prompts(30000, 500)
    assert [len(c) for c in Sizer(16, 1024).chunks(rows, 30000)] == [1]


# -- rubrics ------------------------------------------------------------------


def test_descriptions_are_shown_beside_their_option():
    question = choice(
        "Which queue?",
        ["billing", "abuse"],
        descriptions=["money questions", "reports of misuse"],
    )
    body = messages("a state", question, None, "AB")[-1]["content"]
    assert "A. billing — money questions" in body
    assert "B. abuse — reports of misuse" in body


def test_descriptions_follow_the_option_through_a_relabelling():
    question = choice("Which?", ["a", "b", "c"], descriptions=["first", "second", "third"])
    body = messages("s", question, (2, 0, 1), "XYZ")[-1]["content"]
    assert "X. c — third" in body and "Y. a — first" in body and "Z. b — second" in body


def test_a_question_without_rubrics_renders_bare():
    body = messages("s", choice("Which?", ["a", "b"]), None, "AB")[-1]["content"]
    assert "A. a\nB. b" in body


def test_a_mismatched_rubric_is_refused():
    with pytest.raises(ValueError, match="descriptions"):
        Question("Which?", ("a", "b"), None, ("only one",))


def test_rubrics_survive_coercion_from_a_mapping():
    question = Question.coerce(
        {"text": "Which?", "options": ["a", "b"], "descriptions": ["one", "two"]}
    )
    assert question.descriptions == ("one", "two")
