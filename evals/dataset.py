"""Fetch TypeSafe's public evaluation examples and flatten them into questions.

    uv run evals/dataset.py            # download, verify, convert
    uv run evals/dataset.py --summary  # describe what is already on disk

The source is the public viewer payload behind https://evals.typesafe.ai/ , the
same data [open-jev](https://github.com/JoshuaSP/open-jev) vendored. Nothing is
committed here: the payloads are downloaded, checked against the pinned hashes in
`manifest.json`, and flattened into one question per row.

Read `README.md` in this directory before quoting any number off it. The short
version: these are twenty diagnostic cases chosen to show disagreement, the
reference answers are model-derived rather than human ground truth, and a score
from them is not a reproduction of the published benchmark.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "data"
QUESTIONS = CACHE / "questions.jsonl"


def download(workflow: dict, refresh: bool = False) -> dict:
    """Fetch one viewer payload, verify its hash, and unwrap the JSON inside."""
    raw_path = CACHE / f"{workflow['workflow']}.js"
    if refresh or not raw_path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        # The site refuses urllib's default agent, so identify ourselves plainly.
        request = urllib.request.Request(
            workflow["source"], headers={"User-Agent": "jeff-evals/0.1 (+github.com/jeff)"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw_path.write_bytes(response.read())
    payload = raw_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != workflow["sha256"]:
        raise SystemExit(
            f"{workflow['workflow']}: expected sha256 {workflow['sha256']}, got {digest}.\n"
            "Upstream has been republished. Update manifest.json — both the ?v= in the "
            "URL and the checksum — and note the change alongside any results."
        )
    text = payload.decode("utf-8")
    # The payload is `__VIEWER_DATA__({...});` and wraps the evaluation object
    # alongside the viewer's own icons and glossary.
    return json.loads(text[text.index("(") + 1 : text.rindex(")")])["eval"]


def reference(sets: list[dict] | None) -> str | None:
    """One reference value from the annotation sets, or None if they disagree.

    Most questions carry two reference annotations. A plurality of their chosen
    values settles it. When they split, the annotations' own probability
    distributions break the tie if every one of them has probabilities — they
    agree with the vote wherever both rules apply. Anything still tied is
    dropped rather than resolved arbitrarily: those are questions the references
    themselves disagree about.
    """
    if not sets:
        return None
    votes = collections.Counter(canonical(entry["value"]) for entry in sets)
    (top, count), *rest = votes.most_common()
    if not rest or rest[0][1] != count:
        return top

    pooled: dict[str, float] = collections.defaultdict(float)
    for entry in sets:
        if not entry.get("probabilities"):
            return None
        for option, mass in entry["probabilities"].items():
            pooled[canonical(option)] += mass / len(sets)
    ranked = sorted(pooled.items(), key=lambda item: -item[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0] if ranked else None


def canonical(value) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def options_for(question: dict) -> tuple[list[str], list[str] | None]:
    """The allowed values and their rubric lines, by question type.

    `noul` is a boolean, `score` is an ordinal indexed from zero, and `choice`
    is categorical over named keys. Most carry per-option criteria, which are
    the rubric the model is meant to apply; a few booleans carry none, and go
    to the model on their wording alone.
    """
    criteria = question["criteria"]
    if question["type"] == "noul":
        values = ["false", "true"]
        return values, [criteria[v] for v in values] if criteria else None
    if question["type"] == "score":
        return [str(i) for i in range(len(criteria))], list(criteria)
    return list(criteria), [criteria[key] for key in criteria]


def flatten(workflow: str, data: dict) -> list[dict]:
    """One row per question TypeSafe's own run actually executed."""
    rows = []
    for case_id, case in data["cases"].items():
        saved = case["models"]["typesafe"]
        for node in saved["nodes"]:
            if not node["ran"]:
                continue
            document = json.dumps(data["documents"][node["doc"]], ensure_ascii=False)
            for qid, index in node["questions"].items():
                question = data["questions"][index]
                values, rubric = options_for(question)
                answer = node["answers"][qid]
                if question["type"] == "noul":
                    jev = {"false": 1 - answer["noul"], "true": answer["noul"]}
                else:
                    jev = answer.get("probabilities")
                rows.append(
                    {
                        "id": f"{workflow}/{case_id}/{node['node']}/{qid}",
                        "workflow": workflow,
                        "document_id": f"{workflow}/{node['doc']}",
                        "document": document,
                        "type": question["type"],
                        "instructions": question["instructions"],
                        "options": values,
                        "descriptions": rubric,
                        "reference": reference(
                            (case["reference_answers"].get(node["node"], {}).get(qid) or {}).get("sets")
                        ),
                        "jev_probabilities": {canonical(k): v for k, v in (jev or {}).items()},
                    }
                )
    return rows


def build(refresh: bool = False) -> list[dict]:
    manifest = json.loads((HERE / "manifest.json").read_text())
    rows = []
    for workflow in manifest["workflows"]:
        rows.extend(flatten(workflow["workflow"], download(workflow, refresh)))
    if len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("duplicate question ids; the upstream shape has changed")
    CACHE.mkdir(parents=True, exist_ok=True)
    with QUESTIONS.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load(refresh: bool = False) -> list[dict]:
    if refresh or not QUESTIONS.exists():
        return build(refresh)
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]


def summarise(rows: list[dict]) -> None:
    scored = [row for row in rows if row["reference"] is not None]
    print(f"{len(rows)} questions, {len(scored)} with an agreed reference")
    print(f"{len({row['document_id'] for row in rows})} documents, "
          f"{len({row['workflow'] for row in rows})} workflows\n")
    by_workflow = collections.Counter(row["workflow"] for row in scored)
    by_type = collections.Counter(row["type"] for row in scored)
    widths = collections.Counter(len(row["options"]) for row in scored)
    for name, counts in (("workflow", by_workflow), ("type", by_type)):
        print(f"scored by {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("options per question: " + ", ".join(f"{k}→{v}" for k, v in sorted(widths.items())))
    missing = len(rows) - len(scored)
    print(f"\ndropped {missing} questions with no reference or a tied one")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download the payloads")
    parser.add_argument("--summary", action="store_true", help="describe the data and stop")
    args = parser.parse_args()
    summarise(load(refresh=args.refresh))
    print(f"\nwrote {QUESTIONS}")
