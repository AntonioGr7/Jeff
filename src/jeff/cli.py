"""`jeff` — answer a batch of typed questions about one state, from the shell.

    echo '{"state": "...", "questions": ["Is it urgent?"]}' | jeff
    jeff examples/ticket.json --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .engine import DEFAULT_MODEL, Jeff


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
    parser.add_argument("--temperature", type=float, default=1.0, help="calibration temperature")
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--stats", action="store_true", help="also report timing and tokens saved")
    args = parser.parse_args(argv)

    payload = _load(args.input)
    jeff = Jeff(
        args.model,
        device=args.device,
        temperature=args.temperature,
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
