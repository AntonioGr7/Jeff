"""Temperature scaling — the cheapest honest calibration there is.

The raw option logits are a *conditional* readout: they say which letter the
model would emit, not how often it is right. One scalar fitted on a handful of
labelled examples fixes most of that gap, and it cannot change any decision —
temperature scaling is monotone, so argmax and accuracy are untouched. Only the
confidences move.

    t = fit_temperature([a.logits for a in answers], labels)
    answers = jeff.ask(state, questions, temperature=t)   # or Jeff(temperature=t)
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    steps: int = 200,
) -> float:
    """Return the temperature minimising negative log-likelihood on held-out data.

    `logits` are per-example option logits (`Answer.logits`), `labels` the index
    of the correct option. Rows may have different option counts.
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
    return float(log_t.exp().item())


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
