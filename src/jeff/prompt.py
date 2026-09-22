"""Typed questions, their answers, and the prompt they render to.

The prompt is laid out state-first on purpose: everything the questions share
(system turn + state) comes before anything that differs, so the shared part is
a genuine token prefix and the engine can prefill it exactly once.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# Single-token answer slots. 16 options is already more than a System-1 decision
# should carry, and it keeps the slots inside the safe single-token range.
LETTERS = "ABCDEFGHIJKLMNOP"

SYSTEM = (
    "You are a decision function. Read the state, then answer the question by "
    "choosing exactly one of the listed options. Reply with the single uppercase "
    "letter of your choice and nothing else."
)


@dataclass(frozen=True)
class Question:
    """One typed decision over the state."""

    text: str
    options: tuple[str, ...] = ("yes", "no")
    id: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("question text must be non-empty")
        if not 2 <= len(self.options) <= len(LETTERS):
            raise ValueError(f"a question needs 2..{len(LETTERS)} options, got {len(self.options)}")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"duplicate options in {self.text!r}")

    @property
    def key(self) -> str:
        return self.id or self.text

    @classmethod
    def coerce(cls, value: "Question | str | Mapping[str, Any]") -> "Question":
        if isinstance(value, Question):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            options = value.get("options", ("yes", "no"))
            return cls(text=value["text"], options=tuple(options), id=value.get("id"))
        raise TypeError(f"cannot read a question from {type(value).__name__}")


def boolean(text: str, *, id: str | None = None) -> Question:
    """A yes/no decision. `p(yes)` is the calibrated-ish probability of the claim."""
    return Question(text, ("yes", "no"), id)


def choice(text: str, options: Sequence[str], *, id: str | None = None) -> Question:
    """A categorical decision over named options."""
    return Question(text, tuple(options), id)


def score(text: str, low: int = 1, high: int = 5, *, id: str | None = None) -> Question:
    """An ordinal decision; the expected value is available on the answer."""
    if high - low + 1 > len(LETTERS):
        raise ValueError("score range is wider than the available answer slots")
    return Question(text, tuple(str(n) for n in range(low, high + 1)), id)


@dataclass(frozen=True)
class Answer:
    """The full distribution over a question's options, plus its readout."""

    question: Question
    probabilities: tuple[float, ...]
    logits: tuple[float, ...]

    @property
    def distribution(self) -> dict[str, float]:
        return dict(zip(self.question.options, self.probabilities))

    @property
    def choice(self) -> str:
        return self.question.options[max(range(len(self.probabilities)), key=self.probabilities.__getitem__)]

    @property
    def confidence(self) -> float:
        return max(self.probabilities)

    @property
    def entropy(self) -> float:
        """Natural-log entropy of the distribution; 0 means fully decided."""
        return -sum(p * math.log(p) for p in self.probabilities if p > 0.0)

    @property
    def expected_value(self) -> float:
        """Probability-weighted option value. Only meaningful for numeric options."""
        return sum(p * float(option) for p, option in zip(self.probabilities, self.question.options))

    def __getitem__(self, option: str) -> float:
        return self.distribution[option]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.question.key,
            "question": self.question.text,
            "choice": self.choice,
            "confidence": round(self.confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in self.distribution.items()},
        }

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v:.3f}" for k, v in self.distribution.items())
        return f"Answer({self.question.key!r} -> {self.choice!r} [{body}])"


def render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def messages(state: str, question: Question) -> list[dict[str, str]]:
    """Chat turns for one decision. Shared text strictly precedes per-question text."""
    options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(question.options))
    user = f"<state>\n{state}\n</state>\n\nQuestion: {question.text}\nOptions:\n{options}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def coerce_all(questions: Iterable[Any]) -> list[Question]:
    out = [Question.coerce(q) for q in questions]
    if not out:
        raise ValueError("no questions given")
    keys = [q.key for q in out]
    if len(set(keys)) != len(keys):
        raise ValueError("question ids must be unique")
    return out
