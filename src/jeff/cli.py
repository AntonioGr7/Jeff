"""`jeff` — answer a batch of typed questions about one state, from the shell.

    echo '{"state": "...", "questions": ["Is it urgent?"]}' | jeff
    jeff examples/ticket.json --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .engine import AGGREGATES, DEFAULT_MODEL, Jeff
from .slots import DEFAULT_SLOTS, SLOT_SETS


def _load(path: str | None) -> dict[str, Any]:
    raw = sys.stdin.read() if path in (None, "-") else open(path, encoding="utf-8").read()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or "state" not in payload or "questions" not in payload:
        raise SystemExit("input must be a JSON object with 'state' and 'questions'")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jeff", description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="?", help="JSON file with 'state' and 'questions' (default: stdin)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda, mps, cpu (default: best available)")
    parser.add_argument("--4bit", dest="quantize", action="store_const", const="4bit",
                        help="hold the weights in 4 bits (NF4), for models too big for the card")
    parser.add_argument("--temperature", type=float, default=1.0, help="calibration temperature")
    parser.add_argument(
        "--permutations",
        default="1",
        help="score each question under N cyclic option relabellings and average out "
        "the model's letter bias; 'all' for a full cover (default: 1, no debiasing)",
    )
    parser.add_argument("--aggregate", default="logmean", choices=AGGREGATES)
    parser.add_argument(
        "--slots",
        default=DEFAULT_SLOTS,
        help=f"symbols the options wear: {', '.join(SLOT_SETS)}, or a literal string "
        f"like ABXY (default: {DEFAULT_SLOTS})",
    )
    parser.add_argument(
        "--shuffle-slots",
        action="store_true",
        help="draw each question its own symbols from the pool, seeded by the question",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.0,
        help="withhold an answer whose winning option scores below this",
    )
    parser.add_argument(
        "--min-mass",
        type=float,
        default=0.0,
        help="withhold an answer when the options captured less than this share "
        "of the model's full next-token distribution",
    )
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--stats", action="store_true", help="also report timing and tokens saved")
    args = parser.parse_args(argv)

    payload = _load(args.input)
    jeff = Jeff(
        args.model,
        device=args.device,
        quantize=args.quantize,
        temperature=args.temperature,
        permutations="all" if args.permutations == "all" else int(args.permutations),
        aggregate=args.aggregate,
        slots=args.slots,
        shuffle_slots=args.shuffle_slots,
        seed=args.seed,
        min_probability=args.min_probability,
        min_mass=args.min_mass,
        max_batch=args.max_batch,
    )
    answers = jeff.ask(payload["state"], payload["questions"])

    result: dict[str, Any] = {"answers": [a.as_dict() for a in answers]}
    if args.stats:
        result["stats"] = jeff.stats
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
