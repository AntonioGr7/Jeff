"""What the two ideas are each worth, measured on your own machine.

    uv run examples/benchmark.py

Three ways to get the same eight decisions out of the same model:

  generate   one autoregressive JSON answer per question, parsed back into code
  per-question   typed logit readout, but re-reading the state for every question
  jeff       typed logit readout off one shared prefill
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from jeff import Jeff
from jeff.prompt import messages
from triage import QUESTIONS, TICKET

REPEATS = 3


def timed(label: str, fn, baseline: float | None = None) -> float:
    fn()  # warm up kernels and the allocator
    best = min(_once(fn) for _ in range(REPEATS))
    speedup = f"  {baseline / best:5.1f}x" if baseline else ""
    print(f"{label:<14} {best * 1000:8.0f} ms{speedup}")
    return best


def _once(fn) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def generate(jeff: Jeff) -> list[str]:
    """The baseline everyone starts with: ask, generate, parse, hope."""
    out = []
    for question in QUESTIONS:
        text = jeff.tokenizer.apply_chat_template(
            messages(TICKET.strip(), question),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = jeff.tokenizer(text, return_tensors="pt").to(jeff.device)
        with torch.inference_mode():
            reply = jeff.model.generate(**ids, max_new_tokens=8, do_sample=False)
        out.append(jeff.tokenizer.decode(reply[0, ids["input_ids"].shape[1] :]))
    return out


def per_question(jeff: Jeff) -> None:
    """Typed readout, but each question pays for the whole state again."""
    for question in QUESTIONS:
        jeff._prefix = None  # forbid prefill reuse
        jeff.decide(TICKET.strip(), question)


def main() -> None:
    jeff = Jeff()
    state = TICKET.strip()
    print(f"model {jeff.name} on {jeff.device}, {len(QUESTIONS)} decisions, best of {REPEATS}\n")

    slow = timed("generate", lambda: generate(jeff))
    timed("per-question", lambda: per_question(jeff), slow)
    timed("jeff", lambda: (setattr(jeff, "_prefix", None), jeff.ask(state, QUESTIONS)), slow)
    timed("jeff (warm)", lambda: jeff.ask(state, QUESTIONS), slow)

    jeff.ask(state, QUESTIONS)
    print(f"\n{json.dumps(jeff.stats, indent=2)}")


if __name__ == "__main__":
    main()
