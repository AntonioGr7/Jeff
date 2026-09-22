# Jeff

Typed probabilistic decisions from a language model, without generating a single token.

You have a blob of unstructured state — a support ticket, a diff, a transcript — and a
dozen questions about it. The usual answer is a dozen prompts, a dozen generations and a
dozen JSON parsers. Jeff prefills the state **once** and reads all dozen answers straight
out of the next-token distribution, in one forward pass.

```python
from jeff import Jeff, boolean, choice, score

jeff = Jeff()
answers = jeff.ask(ticket, [
    boolean("Is the customer asking for a refund?"),
    boolean("Is the customer threatening a chargeback?"),
    choice("Which queue should this go to?", ["billing", "technical", "sales", "abuse"]),
    score("How urgent is this, from 1 to 5?", 1, 5),
])

answers[0].choice          # 'yes'
answers[0]["yes"]          # 0.997
answers[-1].expected_value # 2.51
```

This is an open replication of the idea behind [TypeSafe's Jev][jev], following
[SemIf][semif]'s demonstration that it works on ordinary open models. Their calibration
data was never released, so calibration here is a temperature you fit yourself.

## Two ideas, and that is all

**1. The state is a radix.** Every question about one state renders to a prompt that
begins with the same tokens: system turn, then state, then — only at the very end — the
question and its options. So the shared part is a real token prefix. Jeff tokenises all
the prompts, takes their longest common token prefix, prefills it once, and runs every
question as a suffix branching off that one KV cache.

The split is on **token ids**, not on text. That matters: it makes the split exact by
construction. The suffix tokens are literally the tail of the full prompt's own
tokenisation, so no boundary token can merge, shift or re-segment — the failure mode that
makes hand-cut chat templates fragile. It also means nothing has to know how the chat
template is built; whatever the prompts happen to share is what gets prefilled, including
the template's own boilerplate.

The suffixes go through in one batched forward, left-padded so every row's last real
token lands at the same index, with `position_ids` continuing from the prefix and the
padding masked out. Each row sees exactly *prefix + its own question*, and nothing else.

**2. The answer is a lookup, not a generation.** Options are laid out as lettered slots
(`A.`, `B.`, …), each verified at load time to be one exact round-trip token. The forward
pass ends with a distribution over the whole vocabulary; Jeff reads the logits at the
option letters and softmaxes over just those. Zero tokens are generated, nothing is
parsed, and a type error is not representable — the model cannot answer off-menu, cannot
ramble, cannot emit malformed JSON. What you get back is the full distribution, not just
the argmax.

## Numbers

Eight decisions over one support ticket, `Qwen/Qwen3-0.6B` in bf16 on an RTX 2050
(`uv run examples/benchmark.py`, best of three):

| | time | |
|---|---|---|
| generate one answer per question | 2743 ms | |
| typed readout, no prefix sharing | 1284 ms | 2.1× |
| Jeff | 245 ms | 11.2× |
| Jeff, state already prefilled | 164 ms | 16.7× |

130 shared tokens prefilled once instead of eight times; 253 suffix tokens total; zero
tokens generated. The gap widens with the size of the state and the number of questions,
which is exactly the shape of real triage, extraction and routing workloads.

## Install

```bash
uv sync                      # torch + transformers
uv run examples/triage.py
```

Default model is `Qwen/Qwen3-0.6B` — small enough for a 4 GB card. Any causal LM whose
tokeniser gives each of `A`–`P` a single token will work; pass it as `Jeff("...")` or
`--model`. Bigger is better if it fits.

From the shell:

```bash
echo '{"state": "the box arrived empty", "questions": ["Is this a refund request?"]}' | uv run jeff --stats
```

## Calibration

The raw probabilities are a *conditional* readout: they say which letter the model would
emit, not how often it is right. Small models are badly overconfident. One scalar,
fitted on however many labelled examples you can scrape together, fixes most of it:

```python
from jeff import fit_temperature

t = fit_temperature([a.logits for a in dev_answers], dev_labels)
jeff = Jeff(temperature=t)
```

Temperature scaling is monotone, so it cannot change a single decision — accuracy is
untouched and only the confidences move. `jeff.calibrate` also has
`expected_calibration_error`, `brier_score` and `negative_log_likelihood` for checking
that it helped. This is the honest, cheap version of what TypeSafe trains for directly
with RLCD; it will not manufacture knowledge the model does not have.

## Accuracy of the shared prefill

`tests/test_engine.py` asserts the contract that makes any of this trustworthy: answers
scored off the shared cache match answers scored one prompt at a time. In float32 they
agree to ~1e-6. In bfloat16 they drift by ~1e-3 in probability, because changing the
batch width changes which kernels and reduction orders the backend picks — precision,
not semantics, and the same effect SemIf reports. If you need reproducibility to the bit,
run float32 and a fixed batch size.

## Layout

| | |
|---|---|
| [src/jeff/engine.py](src/jeff/engine.py) | prefill, cache branching, batched suffix forward, typed readout |
| [src/jeff/prompt.py](src/jeff/prompt.py) | `Question`/`Answer`, the state-first prompt |
| [src/jeff/calibrate.py](src/jeff/calibrate.py) | temperature fitting and calibration metrics |
| [src/jeff/cli.py](src/jeff/cli.py) | `jeff` — JSON in, JSON out |
| [examples/](examples/) | ticket triage, benchmark |

## Credit

- [TypeSafe — Introducing System One models and Jev][jev], for the framing: unstructured
  state in, typed probabilistic decisions out.
- [TheoLeeCJ/SemIf][semif], for showing the approach reproduces on 4B open models, with
  benchmarks and a calibration study.
- [NandhaKishorM/laya][laya], which reaches the same place from the other direction —
  encoder classifiers rather than a decoder's logits.

[jev]: https://typesafe.ai/blog/introducing-system-one-models-and-jev
[semif]: https://github.com/TheoLeeCJ/SemIf
[laya]: https://github.com/NandhaKishorM/laya
