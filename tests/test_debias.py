"""Letter bias: the layouts must be balanced, and pooling must cancel the bias.

Both facts are pure arithmetic, so they are tested without a model. The claim
is narrow and worth stating exactly: if the readout is
`score(option) + bias(letter)`, a full cyclic cover plus `logmean` recovers the
score differences exactly. Nothing here promises to fix a bias that depends on
the option's content as well as its letter.
"""

from __future__ import annotations

import pytest
import torch

from jeff.engine import pool
from jeff.prompt import Answer, Question, cyclic_layouts


def test_one_permutation_is_the_question_as_written():
    assert cyclic_layouts(4, 1) == [(0, 1, 2, 3)]


def test_full_cover_sends_every_option_past_every_letter_once():
    for count in range(2, 9):
        layouts = cyclic_layouts(count, "all")
        assert len(layouts) == count
        for option in range(count):
            letters = sorted(layout.index(option) for layout in layouts)
            assert letters == list(range(count))


def test_partial_cover_is_evenly_spaced_and_distinct():
    layouts = cyclic_layouts(6, 3)
    assert len(layouts) == 3
    assert len(set(layouts)) == 3
    assert [layout[0] for layout in layouts] == [0, 2, 4]


def test_asking_for_more_layouts_than_options_is_capped():
    assert len(cyclic_layouts(3, 99)) == 3


def test_rejects_bad_permutation_counts():
    for bad in (0, -1, "some", 1.5):
        with pytest.raises(ValueError):
            cyclic_layouts(4, bad)


def _readout(scores, bias, layout):
    """What a biased model would emit, un-permuted back to canonical order."""
    shown = torch.tensor([scores[option] for option in layout]) + torch.tensor(bias)
    canonical = torch.empty_like(shown)
    canonical[list(layout)] = shown
    return torch.log_softmax(canonical, dim=-1)


@pytest.mark.parametrize("bias", [[3.0, 0.0, -1.0], [0.0, 0.0, 0.0], [5.0, 4.0, 0.0]])
def test_full_cover_cancels_an_additive_letter_bias_exactly(bias):
    scores = [0.0, 1.0, 2.0]
    layouts = cyclic_layouts(3, "all")
    pooled = pool(torch.stack([_readout(scores, bias, layout) for layout in layouts]))

    gaps = (pooled - pooled[0]).tolist()
    assert gaps == pytest.approx([s - scores[0] for s in scores], abs=1e-5)


def test_the_bias_really_would_have_won_without_pooling():
    """Sanity check on the fixture: A is favoured hard enough to flip the answer."""
    scores, bias = [0.0, 1.0, 2.0], [3.0, 0.0, -1.0]
    as_written = _readout(scores, bias, cyclic_layouts(3, 1)[0])
    assert int(as_written.argmax()) == 0  # letter A wins on bias alone
    pooled = pool(torch.stack([_readout(scores, bias, l) for l in cyclic_layouts(3, "all")]))
    assert int(pooled.argmax()) == 2  # content wins once the letters are averaged out


def test_partial_cover_shrinks_the_bias_without_removing_it():
    scores, bias = [0.0, 0.2, 0.4, 0.6], [4.0, 0.0, 0.0, 0.0]
    errors = []
    for permutations in (1, 2, 4):
        layouts = cyclic_layouts(4, permutations)
        pooled = pool(torch.stack([_readout(scores, bias, l) for l in layouts]))
        gaps = (pooled - pooled[0]).tolist()
        errors.append(max(abs(g - (s - scores[0])) for g, s in zip(gaps, scores)))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] == pytest.approx(0.0, abs=1e-5)


def test_mean_and_logmean_agree_when_there_is_no_bias():
    scores = [0.0, 1.0, 2.0]
    stack = torch.stack([_readout(scores, [0.0] * 3, l) for l in cyclic_layouts(3, "all")])
    assert pool(stack, "mean").softmax(-1).tolist() == pytest.approx(
        pool(stack, "logmean").softmax(-1).tolist(), abs=1e-5
    )


def test_rejects_unknown_aggregate():
    with pytest.raises(ValueError):
        pool(torch.zeros(2, 3), "median")


def _answer(probabilities, samples):
    question = Question("probe", tuple("abc"[: len(probabilities)]))
    return Answer(question, tuple(probabilities), (), tuple(tuple(s) for s in samples))


def test_stability_is_the_share_of_layouts_that_agree():
    # Pooled winner is option 0; two of three layouts picked it themselves.
    answer = _answer([0.5, 0.3, 0.2], [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    assert answer.stability == pytest.approx(2 / 3)


def test_stability_is_one_when_every_layout_agrees():
    answer = _answer([0.6, 0.4], [[0.9, 0.1], [0.55, 0.45], [0.7, 0.3]])
    assert answer.stability == 1.0


def test_one_permutation_claims_nothing():
    assert _answer([0.9, 0.1], []).stability == 1.0
    assert _answer([0.9, 0.1], [[0.9, 0.1]]).stability == 1.0


def test_stability_and_disagreement_are_different_questions():
    """A wide menu can move a long way between layouts and still never change its mind."""
    answer = _answer([0.4, 0.3, 0.3], [[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
    assert answer.stability == 1.0          # both layouts chose option 0
    assert answer.disagreement > 0.4        # but the distributions are far apart
