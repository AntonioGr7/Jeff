"""Temperature scaling: it should fix confidence without touching decisions."""

from __future__ import annotations

import random

import pytest

from jeff.calibrate import (
    brier_score,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
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
