# Evaluating against TypeSafe's public examples

```bash
uv run evals/dataset.py                    # download, verify, flatten
uv run evals/run.py                        # score both configurations
uv run evals/run.py --model Qwen/Qwen3-1.7B --json evals/results/1.7b.json
```

Everything else in this repository argues from mechanism: the shared prefill is exact,
the cyclic cover cancels a slot bias, the mass catches a broken prompt. None of that says
whether the answers are **right**. This does, on the only public data with typed
questions and reference answers attached.

## What the data is

TypeSafe publishes twenty worked cases from its evaluation suite at
<https://evals.typesafe.ai/>. `dataset.py` downloads the viewer payloads, checks them
against the pinned hashes in `manifest.json`, and flattens them into one row per
question. Nothing is committed; the files land in `data/`, which is ignored.

408 questions over 45 documents and 4 workflows. **318 are scoreable** — 54 have no
reference answer, and 36 are dropped because the case's two reference annotations
disagree and nothing in the data breaks the tie. Of the scoreable rows: 213 boolean,
85 categorical, 20 ordinal; options run from 2 to 8.

The same data was first assembled this way by
[open-jev](https://github.com/JoshuaSP/open-jev), which is where this approach to it
comes from. They score 337 rows where we score 318; they resolve 19 tied references we
drop, and we have not reproduced their rule.

## What the numbers are not

**These are twenty diagnostic cases, deliberately chosen to show disagreement.** Each
workflow contributes one case where each of three models differs from the others, one
where all three miss the reference, and one where all three agree. They are not a
representative sample of the 711-case suite, and a score on them is **not** a
reproduction of the published benchmark. Do not present it as one.

**The reference answers are model-derived**, not independent human ground truth. When we
say "accuracy" we mean agreement with those references.

**We replay question nodes only.** TypeSafe's workflows also involve conditional
question rounds and policy computations that decide the final action; none of that is
reproduced here, and no call is made to any TypeSafe API.

The comparison against Jev's own saved answers uses the probabilities stored in the same
payload, on exactly the rows we scored. It is a comparison of saved outputs, not a race
under matched conditions.

## What the harness does

Questions are grouped by document, which is the shape the engine is for: one prefill per
document, every question about it scored off that cache.

Two configurations run by default — `as-written` (`permutations=1`) and `debiased`
(`permutations="all"`) — so the question the rest of the repo cannot answer gets an
answer: does relabelling the options change how often we are *right*, or only how the
numbers look. The runner prints how many answers relabelling fixed and how many it broke,
not just the net.

Calibration is **cross-fitted**: questions are split in two by a hash of their id, a
temperature is fitted on each half and used to score the other, so no question is
calibrated by a temperature that saw it. `fit_temperature` clamps to `[0.05, 50]`; a fit
that runs to the ceiling is reported as such, and means the scores were anti-correlated
with the truth — the likelihood is best served by flattening them away. That is a
verdict on the model, not a number to quote.

Rows are also reported by question type and workflow, with the stability and vocabulary
mass distributions, because an aggregate accuracy on 318 heterogeneous questions hides
more than it shows.

## Reproducing

The `?v=` in each source URL is the first eight hex digits of that payload's own sha256,
so the manifest's checksums are the site's own content addresses. If a download fails
verification, upstream has been republished: update the URL and the checksum together,
and say so alongside any result you publish from it.
