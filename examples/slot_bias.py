"""Which answer symbols is this model fairest with, and does shuffling them help?

    uv run examples/slot_bias.py

Nothing says the options must be labelled A, B, C. Different symbol sets carry
different amounts of bias, so this measures it rather than guessing: score a
batch of questions under a full cyclic cover, then read off how much each symbol
pulls, in logits, independently of what was written next to it.

Every probe below has exactly four options, so every question wears the same
four symbols and the pulls are measured against a common baseline. Comparing a
symbol that only ever appears in five-option questions against one that appears
everywhere would not mean anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jeff import Jeff, SLOT_SETS, choice, estimate_slot_bias, spread
from triage import TICKET

WIDTH = 4
PROBES = [
    choice("Which queue should this go to?", ["billing", "technical", "sales", "abuse"], id="queue"),
    choice("What is the root cause?", ["failed cancellation", "duplicate charge", "user error", "unclear"], id="cause"),
    choice("What tone should the reply take?", ["apologetic", "neutral", "firm", "formal"], id="tone"),
    choice("When should we reply?", ["within an hour", "today", "this week", "whenever"], id="when"),
    choice("Who should handle this?", ["a bot", "support", "a manager", "finance"], id="owner"),
    choice("How urgent is this?", ["not at all", "somewhat", "very", "critical"], id="urgency"),
    choice("What is the risk if we ignore it?", ["none", "a complaint", "a chargeback", "churn"], id="risk"),
    choice("What does the customer want first?", ["money back", "an apology", "an explanation", "a call"], id="wants"),
    choice("How clear is the request?", ["unclear", "vague", "clear", "very clear"], id="clarity"),
    choice("How long has this been going on?", ["days", "weeks", "months", "unknown"], id="age"),
]


def agreement(a, b) -> int:
    return sum(x.choice == y.choice for x, y in zip(a, b))


def main() -> None:
    jeff = Jeff()
    state = TICKET.strip()
    print(f"{len(PROBES)} questions, {WIDTH} options each, full cyclic cover\n")

    print(f"{'set':<10} {'symbols':<10} {'spread':>7}   pull per symbol, in logits")
    print("-" * 74)
    measured = {}
    for name, pool in SLOT_SETS.items():
        if len(pool) < WIDTH:
            continue
        answers = jeff.ask(state, PROBES, permutations="all", slots=pool[:WIDTH])
        bias = estimate_slot_bias(answers)
        measured[name] = bias
        shown = "  ".join(f"{s}{v:+.2f}" for s, v in sorted(bias.items(), key=lambda kv: -kv[1]))
        print(f"{name:<10} {pool[:WIDTH]:<10} {spread(bias):7.2f}   {shown}")

    fairest = min(measured, key=lambda name: spread(measured[name]))
    worst = max(measured, key=lambda name: spread(measured[name]))
    print(
        f"\nFairest here: {fairest!r} at {spread(measured[fairest]):.2f} logits of spread; "
        f"worst {worst!r} at {spread(measured[worst]):.2f}."
    )

    # The whole pool at once. Shuffled subsets make every question's baseline
    # converge on the pool mean, which is what makes 16 symbols comparable.
    wide = jeff.ask(state, PROBES, permutations="all", slots="letters", shuffle_slots=True)
    bias = estimate_slot_bias(wide)
    ranked = sorted(bias.items(), key=lambda kv: -kv[1])
    print(f"\nAcross all of 'letters' ({len(bias)} symbols seen, shuffled subsets):")
    print("  most pull   " + "  ".join(f"{s}{v:+.2f}" for s, v in ranked[:5]))
    print("  least pull  " + "  ".join(f"{s}{v:+.2f}" for s, v in ranked[-5:]))

    # Does any of this rescue the cheap path, where nothing cancels?
    truth = jeff.ask(state, PROBES, permutations="all")
    print(f"\nAgreement with the debiased answer, out of {len(PROBES)}:")
    for label, kwargs in (
        ("A, B, C, D", dict(permutations=1)),
        ("shuffled", dict(permutations=1, shuffle_slots=True)),
        (f"fairest set ({fairest})", dict(permutations=1, slots=SLOT_SETS[fairest][:WIDTH])),
        ("2 permutations", dict(permutations=2)),
    ):
        print(f"  {label:<22} {agreement(jeff.ask(state, PROBES, **kwargs), truth)}")


if __name__ == "__main__":
    main()
