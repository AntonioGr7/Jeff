"""Temperature scaling — the cheapest honest calibration there is.

The raw option logits are a *conditional* readout: they say which letter the
model would emit, not how often it is right. One scalar fitted on a handful of
labelled examples fixes most of that gap, and it cannot change any decision —
temperature scaling is monotone, so argmax and accuracy are untouched. Only the
confidences move.

    t = fit_temperature([a.logits for a in answers], labels)
    answers = jeff.ask(state, questions, temperature=t)   # or Jeff(temperature=t)

Calibration says whether a 0.9 means 0.9. `risk_coverage` asks the question you
deploy on: if you only act above some confidence, how much do you get to act
on, and how often is it wrong?
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


BOUNDS = (0.05, 50.0)


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    steps: int = 200,
    bounds: tuple[float, float] = BOUNDS,
) -> float:
    """Return the temperature minimising negative log-likelihood on held-out data.

    `logits` are per-example option logits (`Answer.logits`), `labels` the index
    of the correct option. Rows may have different option counts.

    The result is clamped to `bounds`. A fit that runs to the ceiling is not a
    number, it is a verdict: the scores were anti-correlated with the truth, so
    the likelihood is best served by flattening them to nothing. Compare the
    result against `bounds` before quoting it.
    """
    if len(logits) != len(labels):
        raise ValueError("need one label per row")
    if not logits:
        raise ValueError("need at least one labelled row")
    for row, label in zip(logits, labels):
        if not 0 <= label < len(row):
            raise ValueError(f"label {label} is outside the {len(row)} options of its row")

    rows = [torch.tensor(row, dtype=torch.float64) for row in logits]
    targets = torch.tensor(list(labels))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        temperature = log_t.exp()
        loss = torch.stack(
            [
                -torch.log_softmax(row / temperature, dim=-1)[target]
                for row, target in zip(rows, targets)
            ]
        ).mean()
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(min(max(log_t.exp().item(), bounds[0]), bounds[1]))


def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> float:
    """ECE of the top-1 confidence: mean gap between confidence and accuracy."""
    if len(probabilities) != len(labels):
        raise ValueError("need one label per row")
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for row, label in zip(probabilities, labels):
        best = max(range(len(row)), key=row.__getitem__)
        confidence = row[best]
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, float(best == label)))
    total = len(labels)
    return sum(
        len(bucket) / total * abs(
            sum(c for c, _ in bucket) / len(bucket) - sum(h for _, h in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


def brier_score(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Multiclass Brier score; lower is better, and it rewards honest spread."""
    if len(probabilities) != len(labels):
        raise ValueError("need one label per row")
    total = 0.0
    for row, label in zip(probabilities, labels):
        total += sum((p - (i == label)) ** 2 for i, p in enumerate(row))
    return total / len(labels)


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int]
) -> float:
    return -sum(
        math.log(max(row[label], 1e-12)) for row, label in zip(probabilities, labels)
    ) / len(labels)


def risk_coverage(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int]
) -> list[tuple[float, float, float]]:
    """The selective-prediction curve: `(threshold, coverage, risk)` at every cut.

    Answer only when the top probability is at least `threshold` and abstain
    otherwise; `coverage` is the share answered and `risk` the error rate on
    that share. `threshold` is exactly what `min_probability` expects.

    Tied confidences stay together. No threshold can answer one of two rows at
    0.9998 and withhold the other, and a saturated model produces a lot of
    those, so cutting between them would report a coverage you cannot deploy.
    """
    if len(probabilities) != len(labels):
        raise ValueError("need one label per row")
    if not labels:
        raise ValueError("need at least one labelled row")
    scored = []
    for row, label in zip(probabilities, labels):
        best = max(range(len(row)), key=row.__getitem__)
        scored.append((row[best], best == label))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    curve, wrong = [], 0
    for index, (confidence, hit) in enumerate(scored):
        wrong += not hit
        if index + 1 == len(scored) or scored[index + 1][0] < confidence:
            kept = index + 1
            curve.append((confidence, kept / len(scored), wrong / kept))
    return curve


def coverage_at_risk(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int], risk: float = 0.05
) -> float:
    """Largest share of answers a single threshold can keep at or below `risk`.

    The number that says how much traffic you could automate. Accuracy can move
    a few points while this moves tenfold, because it depends on whether the
    confident answers are the right ones, not on how many are right overall.
    Zero when even the most confident group is wrong too often.
    """
    return max((c for _, c, r in risk_coverage(probabilities, labels) if r <= risk), default=0.0)


def area_under_risk_coverage(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int]
) -> float:
    """AURC: the risk averaged over every coverage; lower is better.

    `coverage_at_risk` reads one point of the curve, this summarises all of it,
    so it does not hinge on where you put the target. Its floor is not zero: a
    perfect ranking still pays for the errors it has to answer at full coverage.
    """
    area, previous = 0.0, 0.0
    for _, coverage, risk in risk_coverage(probabilities, labels):
        area += risk * (coverage - previous)
        previous = coverage
    return area
