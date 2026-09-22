"""Confidence over a menu nobody read.

    uv run examples/vocabulary_mass.py

`probabilities` is a softmax over a handful of vocabulary entries out of a
hundred thousand: a preference *within the menu*. It cannot tell you whether the
model was answering your question at all. `mass` can — it is how much of the
model's real next-token distribution landed on your options.

Below, the same questions are scored twice. The second run leaves Qwen's
thinking mode on, so the model is actually about to emit `<think>`. It still has
a favourite letter, so the confidences stay high and look perfectly healthy.
Only the mass gives the game away.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jeff import Jeff
from jeff.prompt import messages
from triage import QUESTIONS, TICKET


def rebuild(jeff: Jeff, **template):
    """The prompt built wrongly, the way a caller might build it by accident."""

    def encode(state, question, layout=None, slots=""):
        text = jeff.tokenizer.apply_chat_template(
            messages(state, question, layout, slots), tokenize=False, **template
        )
        return jeff.tokenizer.encode(text, add_special_tokens=False)

    return encode


def report(label: str, answers) -> None:
    print(f"\n{label}")
    for answer in answers:
        flag = "  <-- withheld" if answer.abstained else ""
        print(
            f"  {answer.question.key:<16} {answer.choice:<20} "
            f"confidence {answer.confidence:6.1%}   mass {answer.mass:.6f}{flag}"
        )


def main() -> None:
    jeff = Jeff(min_mass=0.5)
    state = TICKET.strip()

    report("as built (thinking off, generation prompt on):", jeff.ask(state, QUESTIONS[:5]))

    for label, template in (
        ("thinking left on:", dict(add_generation_prompt=True, enable_thinking=True)),
        ("no generation prompt:", dict(add_generation_prompt=False)),
    ):
        jeff._encode, jeff._prefix = rebuild(jeff, **template), None
        report(label, jeff.ask(state, QUESTIONS[:5]))

    print(
        "\nEvery confidence in those last two blocks is a renormalisation artifact.\n"
        "Without the mass they are indistinguishable from the healthy run — which is\n"
        "the whole argument for reporting it. `Jeff(min_mass=0.5)` withholds them."
    )


if __name__ == "__main__":
    main()
