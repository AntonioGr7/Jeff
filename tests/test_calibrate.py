"""Temperature scaling: it should fix confidence without touching decisions."""

from __future__ import annotations

import random

import pytest

from jeff.calibrate import (
    area_under_risk_coverage,
    brier_score,
    coverage_at_risk,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    risk_coverage,
)


def overconfident(seed: int = 0, n: int = 400):
    """Logits that are right ~75% of the time but scream 99%."""
    rng = random.Random(seed)
    logits, labels = [], []
    for _ in range(n):
        correct = rng.random() < 0.75
        margin = rng.uniform(4.0, 8.0)
        logits.append([margin, 0.0])
        labels.append(0 if correct else 1)
    return logits, labels


def softmax(row, t):
    import math

    scaled = [v / t for v in row]
    top = max(scaled)
    exp = [math.exp(v - top) for v in scaled]
    total = sum(exp)
    return [e / total for e in exp]


def test_temperature_improves_calibration_without_changing_decisions():
    logits, labels = overconfident()
    t = fit_temperature(logits, labels)
    assert t > 1.0  # an overconfident model needs softening

    before = [softmax(row, 1.0) for row in logits]
    after = [softmax(row, t) for row in logits]

    assert [max(range(2), key=r.__getitem__) for r in before] == [
        max(range(2), key=r.__getitem__) for r in after
    ]
    assert expected_calibration_error(after, labels) < expected_calibration_error(before, labels)
    assert brier_score(after, labels) < brier_score(before, labels)
    assert negative_log_likelihood(after, labels) < negative_log_likelihood(before, labels)


def test_fit_handles_ragged_option_counts():
    logits = [[2.0, 0.0], [1.0, 0.5, 0.0], [3.0, 1.0, 0.0, 0.0]]
    assert fit_temperature(logits, [0, 0, 0]) > 0.0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_temperature([[1.0, 0.0]], [0, 1])
    with pytest.raises(ValueError):
        fit_temperature([[1.0, 0.0]], [5])
    with pytest.raises(ValueError):
        fit_temperature([], [])


def test_coverage_follows_the_confident_answers():
    # Four right answers on top, then a wrong one, then a right one below it.
    probabilities = [[0.99, 0.01], [0.95, 0.05], [0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.6, 0.4]]
    labels = [0, 0, 0, 0, 1, 0]
    curve = risk_coverage(probabilities, labels)
    assert [t for t, _, _ in curve] == [0.99, 0.95, 0.9, 0.85, 0.8, 0.6]
    assert curve[3][1:] == (4 / 6, 0.0)
    assert curve[4][2] == pytest.approx(1 / 5)
    assert coverage_at_risk(probabilities, labels, 0.0) == pytest.approx(4 / 6)
    assert coverage_at_risk(probabilities, labels, 0.2) == 1.0  # 1 wrong of 6 is under 20%
    assert coverage_at_risk(probabilities, labels, 0.1) == pytest.approx(4 / 6)


def test_tied_confidences_are_never_split():
    # Right and wrong at the same confidence: no threshold keeps only the right one.
    probabilities = [[0.9, 0.1], [0.9, 0.1], [0.7, 0.3]]
    labels = [0, 1, 0]
    curve = risk_coverage(probabilities, labels)
    assert [(t, c) for t, c, _ in curve] == [(0.9, pytest.approx(2 / 3)), (0.7, 1.0)]
    assert coverage_at_risk(probabilities, labels, 0.05) == 0.0


def test_aurc_rewards_ranking_not_accuracy():
    # Same answers, same accuracy; only which one the model is sure about differs.
    labels = [0, 0, 1]
    good = [[0.9, 0.1], [0.8, 0.2], [0.6, 0.4]]   # the wrong answer is the least confident
    bad = [[0.6, 0.4], [0.8, 0.2], [0.9, 0.1]]    # the wrong answer is the most confident
    assert area_under_risk_coverage(good, labels) < area_under_risk_coverage(bad, labels)
    assert area_under_risk_coverage(good, labels) == pytest.approx((0 + 0 + 1 / 3) / 3)
    assert area_under_risk_coverage([[0.9, 0.1]] * 3, [0, 0, 0]) == 0.0


def test_coverage_threshold_matches_min_probability():
    # The curve's threshold is inclusive, as Jeff's abstention floor is.
    probabilities = [[0.9, 0.1], [0.7, 0.3], [0.55, 0.45]]
    labels = [0, 0, 1]
    threshold, coverage, _ = risk_coverage(probabilities, labels)[1]
    answered = [row for row in probabilities if not max(row) < threshold]
    assert len(answered) / len(probabilities) == coverage
