"""Answer slots: the symbols the options wear, and what they cost you.

Nothing requires the slots to be `A`, `B`, `C`. They only have to be single
tokens the model will happily emit. Which symbols you pick changes the bias you
are fighting, so this module makes the set a choice — and gives you an estimator
so the choice can be made from measurements rather than from folklore.

Note what randomising the symbols does and does not do. It does not make any
single answer unbiased; it stops the *same* option from being the favoured one
across a whole workload, by turning a systematic bias into variance. Only the
cyclic cover in `jeff.engine` actually cancels. The two compose because the
symbol subset is drawn once per question and the cover runs inside it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable

SLOT_SETS: dict[str, str] = {
    "letters": "ABCDEFGHIJKLMNOP",
    "digits": "0123456789",
    "extended": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "lower": "abcdefghijklmnop",
    "roman": "IVXLCDM",
}

DEFAULT_SLOTS = "letters"


def resolve_slots(spec: str) -> str:
    """A named preset, or a literal string of symbols to use verbatim."""
    pool = SLOT_SETS.get(spec, spec)
    if len(pool) < 2:
        raise ValueError(f"a slot pool needs at least two symbols, got {pool!r}")
    if len(set(pool)) != len(pool):
        raise ValueError(f"slot pool {pool!r} repeats a symbol")
    return pool


def take_slots(pool: str, count: int, *, key: str | None = None, seed: int = 0) -> str:
    """Pick `count` symbols from `pool`.

    With no `key`, the first `count` in order — the conventional A, B, C. With a
    `key` (use the question's), a subset drawn and shuffled deterministically
    from it, so the same question always gets the same symbols and answers stay
    reproducible, while different questions stop sharing one favoured slot.
    """
    if count > len(pool):
        raise ValueError(
            f"{count} options need {count} slot symbols but pool {pool!r} has {len(pool)}"
            + ("; digits only go up to ten" if pool == SLOT_SETS["digits"] else "")
        )
    if key is None:
        return pool[:count]
    return "".join(random.Random(f"{seed}:{key}").sample(pool, count))


def estimate_slot_bias(answers: Iterable[Any]) -> dict[str, float]:
    """Estimate each symbol's pull, in logits, from a full-cover run.

    Pass answers from `ask(..., permutations="all")`. Writing the readout as
    `score(option) + bias(symbol)`, every option passes every symbol exactly
    once, so centring each layout and then averaging over layouts leaves the
    score term as a constant and the symbol term standing. Results are centred
    on zero: positive means the model leans toward that symbol regardless of
    what is written next to it.

    Each question is centred on *its own* symbol set, so symbols are only
    strictly comparable when they co-occur. Compare like with like: either give
    every question the same number of options, or use `shuffle_slots=True` over
    a pool, which makes every question's baseline converge on the pool mean.
    Mixing widths silently compares a symbol that only appears in the wide
    questions against one that appears everywhere.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for answer in answers:
        layouts, samples, slots = answer.layouts, answer.samples, answer.slots
        if not layouts or len(layouts) != len(answer.question.options):
            continue  # only a full cover identifies the symbol term
        for layout, sample in zip(layouts, samples):
            row = [math.log(max(sample[option], 1e-12)) for option in layout]
            centre = sum(row) / len(row)
            for symbol, value in zip(slots, row):
                totals[symbol] = totals.get(symbol, 0.0) + value - centre
                counts[symbol] = counts.get(symbol, 0) + 1
    if not counts:
        raise ValueError("need answers scored with permutations='all'")
    bias = {symbol: totals[symbol] / counts[symbol] for symbol in totals}
    centre = sum(bias.values()) / len(bias)
    return {symbol: value - centre for symbol, value in sorted(bias.items())}


def spread(bias: dict[str, float]) -> float:
    """How unfair a symbol set is overall, in logits: best symbol minus worst."""
    return max(bias.values()) - min(bias.values()) if bias else 0.0
