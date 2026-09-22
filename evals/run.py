"""Score Jeff against TypeSafe's public evaluation examples.

    uv run evals/run.py                              # both configurations
    uv run evals/run.py --model Qwen/Qwen3-1.7B
    uv run evals/run.py --config debiased --json results.json

Questions are grouped by document, which is also the shape the engine is built
for: one prefill per document, every question about it scored off that cache.

What this measures is whether the machinery in this repo earns its keep —
whether relabelling the options improves *accuracy* and not just theory, and
whether one fitted temperature makes the confidences mean anything. Read
`README.md` here for what the numbers are and are not.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset import load

from jeff import Jeff, Question
from jeff.calibrate import (
    BOUNDS,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
)

CONFIGS = {
    "as-written": dict(permutations=1),
    "debiased": dict(permutations="all"),
}


def ask_all(jeff: Jeff, rows: list[dict], **settings) -> list[dict]:
    """Run every question, one prefill per document, and keep the raw readout."""
    by_document: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)

    results, started, done = [], time.perf_counter(), 0
    for document_id, group in by_document.items():
        questions = [
            Question(
                text=row["instructions"],
                options=tuple(row["options"]),
                id=row["id"],
                descriptions=tuple(row["descriptions"]) if row["descriptions"] else None,
            )
            for row in group
        ]
        answers = jeff.ask(group[0]["document"], questions, **settings)
        for row, answer in zip(group, answers):
            results.append(
                {
                    "id": row["id"],
                    "workflow": row["workflow"],
                    "type": row["type"],
                    "options": row["options"],
                    "reference": row["reference"],
                    "predicted": answer.choice,
                    "correct": answer.choice == row["reference"],
                    "probabilities": answer.distribution,
                    "logits": list(answer.logits),
                    "label": row["options"].index(row["reference"]),
                    "confidence": answer.confidence,
                    "mass": answer.mass,
                    "stability": answer.stability,
                    "jev_probabilities": row["jev_probabilities"],
                }
            )
        done += len(group)
        print(f"\r  {done}/{len(rows)} questions, {len(results)} scored", end="", file=sys.stderr)
    print(f"  ({time.perf_counter() - started:.0f}s)", file=sys.stderr)
    return results


def metrics(results: list[dict], temperature: float | None = None) -> dict:
    """Accuracy plus the three proper-scoring numbers, optionally recalibrated."""
    if temperature:
        rows = [_rescale(result, temperature) for result in results]
    else:
        rows = [result["probabilities"] for result in results]
    probabilities = [[row[option] for option in r["options"]] for row, r in zip(rows, results)]
    labels = [result["label"] for result in results]
    correct = [
        max(row, key=row.__getitem__) == result["reference"]
        for row, result in zip(rows, results)
    ]
    return {
        "n": len(results),
        "accuracy": sum(correct) / len(correct),
        "brier": brier_score(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels),
        "nll": negative_log_likelihood(probabilities, labels),
    }


def _rescale(result: dict, temperature: float) -> dict[str, float]:
    import math

    scaled = [value / temperature for value in result["logits"]]
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    return {option: w / total for option, w in zip(result["options"], weights)}


def cross_fit_temperature(results: list[dict]) -> tuple[float, dict]:
    """Fit on one half, score the other, both ways round.

    Fitting and reporting a temperature on the same rows would flatter it. Each
    question is calibrated by a temperature that never saw it.
    """
    folds = {0: [], 1: []}
    for result in results:
        # Not hash(): that is salted per process, and the split must not be.
        digest = hashlib.sha256(result["id"].encode()).digest()
        folds[digest[0] % 2].append(result)
    temperatures, scored = {}, []
    for fold in (0, 1):
        train, test = folds[1 - fold], folds[fold]
        if not train or not test:
            continue
        temperature = fit_temperature(
            [row["logits"] for row in train], [row["label"] for row in train]
        )
        temperatures[fold] = temperature
        for result in test:
            scored.append({**result, "probabilities": _rescale(result, temperature)})
    mean = sum(temperatures.values()) / len(temperatures)
    return mean, metrics(scored)


def breakdown(results: list[dict], key: str) -> dict[str, tuple[int, float]]:
    groups = collections.defaultdict(list)
    for result in results:
        groups[result[key]].append(result["correct"])
    return {k: (len(v), sum(v) / len(v)) for k, v in sorted(groups.items())}


def against_jev(results: list[dict]) -> dict | None:
    """How Jev's own saved answers do on exactly the rows we scored."""
    usable = [r for r in results if r["jev_probabilities"]]
    if not usable:
        return None
    jev_correct = agree = 0
    for result in usable:
        jev = max(result["jev_probabilities"], key=result["jev_probabilities"].get)
        jev_correct += jev == result["reference"]
        agree += jev == result["predicted"]
    return {
        "n": len(usable),
        "jev_accuracy": jev_correct / len(usable),
        "ours_accuracy": sum(r["correct"] for r in usable) / len(usable),
        "agreement": agree / len(usable),
    }


def line(label: str, m: dict) -> str:
    return (f"  {label:<24} {m['accuracy']:7.1%} {m['brier']:8.3f} "
            f"{m['ece']:7.3f} {m['nll']:7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", choices=[*CONFIGS, "both"], default="both")
    parser.add_argument("--workflow", action="append", help="limit to these workflows")
    parser.add_argument("--limit", type=int, help="first N questions, for a smoke run")
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=12288,
                        help="KV budget per batch; lower it if a long document runs you out")
    parser.add_argument("--json", help="write the full per-question record here")
    args = parser.parse_args()

    rows = [row for row in load() if row["reference"] is not None]
    if args.workflow:
        rows = [row for row in rows if row["workflow"] in args.workflow]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no questions selected")

    jeff = Jeff(
        *([args.model] if args.model else []),
        max_batch=args.max_batch,
        max_batch_tokens=args.max_batch_tokens,
    )
    documents = len({row["document_id"] for row in rows})
    print(f"{len(rows)} questions over {documents} documents, model {jeff.name}\n")
    print(f"  {'configuration':<24} {'accuracy':>7} {'brier':>8} {'ECE':>7} {'NLL':>7}")

    names = list(CONFIGS) if args.config == "both" else [args.config]
    everything = {}
    for name in names:
        print(f"\n{name}:", file=sys.stderr)
        results = ask_all(jeff, rows, **CONFIGS[name])
        everything[name] = results
        print(line(name, metrics(results)))
        temperature, calibrated = cross_fit_temperature(results)
        pegged = " (at bound: the scores carry no usable confidence)" if temperature >= BOUNDS[1] else ""
        print(line(f"  + temperature {temperature:.2f}", calibrated) + pegged)

    for name, results in everything.items():
        print(f"\n{name}, accuracy by question type and workflow:")
        for key in ("type", "workflow"):
            parts = [f"{k} {acc:.1%} (n={n})" for k, (n, acc) in breakdown(results, key).items()]
            print(f"  {key:<9} " + "   ".join(parts))
        stability = [r["stability"] for r in results]
        shaky = [r for r in results if r["stability"] < 0.6]
        print(f"  stability mean {sum(stability) / len(stability):.2f}; "
              f"{len(shaky)} answers below 0.6, of which {sum(r['correct'] for r in shaky)} correct")
        masses = [r["mass"] for r in results]
        print(f"  vocabulary mass min {min(masses):.4f}, mean {sum(masses) / len(masses):.4f}")

    if "as-written" in everything and "debiased" in everything:
        before = {r["id"]: r["correct"] for r in everything["as-written"]}
        after = {r["id"]: r["correct"] for r in everything["debiased"]}
        fixed = sum(1 for k in before if after[k] and not before[k])
        broken = sum(1 for k in before if before[k] and not after[k])
        print(f"\nrelabelling fixed {fixed} answers and broke {broken}, "
              f"net {fixed - broken:+d} of {len(before)}")

    comparison = against_jev(next(iter(everything.values())))
    if comparison:
        print(f"\nOn the {comparison['n']} rows where Jev's saved answer is available: "
              f"Jev {comparison['jev_accuracy']:.1%}, this {comparison['ours_accuracy']:.1%}, "
              f"agreeing with each other {comparison['agreement']:.1%} of the time.")

    print("\nTwenty diagnostic cases with model-derived references; see evals/README.md.")
    if args.json:
        Path(args.json).write_text(json.dumps(everything, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
