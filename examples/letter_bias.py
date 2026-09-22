"""Does the model answer the question, or answer the letter?

    uv run examples/letter_bias.py

Scores each question under every cyclic relabelling of its options. If the
readout were purely about content, all layouts would agree exactly. The gap is
the model's letter bias, and the last column is what averaging them removes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jeff import Jeff
from triage import QUESTIONS, TICKET


def main() -> None:
    jeff = Jeff()
    state = TICKET.strip()

    naive = jeff.ask(state, QUESTIONS)
    fair = jeff.ask(state, QUESTIONS, permutations="all")

    print(f"{'question':<16} {'as written':<22} {'debiased':<22} {'moved':>6} {'flip'}")
    print("-" * 76)
    flips = 0
    for before, after in zip(naive, fair):
        flipped = before.choice != after.choice
        flips += flipped
        print(
            f"{before.question.key:<16} "
            f"{before.choice + ' ' + format(before.confidence, '.1%'):<22} "
            f"{after.choice + ' ' + format(after.confidence, '.1%'):<22} "
            f"{after.disagreement:6.3f} {'  <-- flipped' if flipped else ''}"
        )

    worst = max(fair, key=lambda a: a.disagreement)
    print(f"\n{flips} of {len(fair)} answers changed once the letters stopped mattering.")
    print(
        f"Most layout-sensitive: {worst.question.key!r}, "
        f"total-variation spread {worst.disagreement:.3f} across "
        f"{worst.permutations} layouts."
    )

    print(f"\nHow {worst.question.key!r} reads under each layout:")
    for layout, sample in zip(worst.layouts, worst.samples):
        shown = ", ".join(
            f"{option}={p:.1%}" for option, p in zip(worst.question.options, sample)
        )
        first = worst.question.options[layout[0]]
        print(f"  {worst.slots[0]}={first!r:<22} {shown}")

    print(f"\n{jeff.stats['forward_rows']} scored rows, one prefill, 0 tokens generated.")


if __name__ == "__main__":
    main()
