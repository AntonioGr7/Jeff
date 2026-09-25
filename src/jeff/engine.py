"""The engine: prefill the shared state once, score every question in parallel.

Two ideas, and that is the whole trick.

1. *Radix prefill.* Every question about one state renders to a prompt that
   starts with the same tokens. We tokenise them all, take the longest common
   token prefix, and run it through the model exactly once. Splitting on token
   ids rather than on text means the split is exact by construction: the suffix
   tokens are literally the tail of the full prompt's own tokenisation, so no
   boundary token can merge or shift.

2. *Typed readout.* We never generate. One forward pass over the suffixes
   returns the next-token distribution at the end of each prompt; we read the
   logits of the option letters, and softmax over just those. Zero output
   tokens, no parsing, and a type error is not representable.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .prompt import (
    Answer,
    Question,
    coerce_all,
    cyclic_layouts,
    messages,
    render_state,
)
from .slots import DEFAULT_SLOTS, resolve_slots, take_slots

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
AGGREGATES = ("logmean", "mean")
QUANTIZE = (None, "4bit")


def _load_model(model: str, device: torch.device, dtype: torch.dtype, quantize: str | None):
    """The model, either as stored or with its linear layers held in 4 bits.

    `4bit` is bitsandbytes NF4 with double quantisation: weights stay packed on
    the card and are unpacked per matmul into `dtype`, which is what fits a 4B
    into 4 GB. It is *not* a lossless speed-up. The logits move, so the answers
    can too, and a quantised model deserves its own eval run before its
    probabilities are trusted.
    """
    if quantize not in QUANTIZE:
        raise ValueError(f"quantize must be one of {QUANTIZE}")
    kwargs: dict[str, Any] = dict(dtype=dtype, attn_implementation="sdpa")
    if quantize is None:
        return AutoModelForCausalLM.from_pretrained(model, **kwargs).to(device).eval()
    if device.type != "cuda":
        raise ValueError("quantize='4bit' needs a CUDA device")
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise ImportError("quantize='4bit' needs bitsandbytes: uv sync --extra quant") from error
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    return AutoModelForCausalLM.from_pretrained(
        model, quantization_config=config, device_map={"": device}, **kwargs
    ).eval()


def _pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _layer_kv(cache: DynamicCache, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one layer's keys/values across transformers' two cache layouts."""
    if hasattr(cache, "layers"):
        layer = cache.layers[index]
        return layer.keys, layer.values
    return cache.key_cache[index], cache.value_cache[index]


def _cache_depth(cache: DynamicCache) -> int:
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def pool(samples: torch.Tensor, aggregate: str = "logmean") -> torch.Tensor:
    """Fold one question's per-layout log-probabilities into a single score.

    `logmean` averages in log space. Write the model's readout as
    `score(option) + bias(letter)`: under a full cyclic cover every option wears
    every letter exactly once, so the average picks up `mean(bias)` — the same
    constant for every option — and that constant vanishes in the softmax. The
    letter bias is gone, exactly, without ever estimating it.

    `mean` averages the probabilities instead. It only marginalises over
    layouts rather than cancelling the bias, but it is the gentler of the two:
    log space is unforgiving, where one layout putting an option near zero
    holds the whole average down.
    """
    if aggregate == "logmean":
        return samples.mean(dim=0)
    if aggregate == "mean":
        return samples.exp().mean(dim=0).log()
    raise ValueError(f"aggregate must be one of {AGGREGATES}")


def _common_prefix_len(sequences: Sequence[Sequence[int]]) -> int:
    """Longest shared token prefix, always leaving every suffix at least one token."""
    limit = min(len(s) for s in sequences) - 1
    first = sequences[0]
    for i in range(limit):
        token = first[i]
        if any(s[i] != token for s in sequences[1:]):
            return i
    return max(limit, 0)


class Jeff:
    """A typed decision function backed by a causal LM.

    >>> jeff = Jeff()
    >>> answers = jeff.ask("The payment failed twice.", ["Is the user blocked?"])
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        quantize: str | None = None,
        temperature: float = 1.0,
        permutations: int | str = 1,
        aggregate: str = "logmean",
        slots: str = DEFAULT_SLOTS,
        shuffle_slots: bool = False,
        seed: int = 0,
        min_probability: float = 0.0,
        min_mass: float = 0.0,
        max_batch: int = 16,
        max_batch_tokens: int = 16384,
    ) -> None:
        self.device = _pick_device(device)
        if dtype is None:
            dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = _load_model(model, self.device, dtype, quantize)
        self.name = model
        self.quantize = quantize
        self.temperature = temperature
        self.permutations = permutations
        self.aggregate = aggregate
        self.slots = slots
        self.shuffle_slots = shuffle_slots
        self.seed = seed
        self.min_probability = min_probability
        self.min_mass = min_mass
        self.max_batch = max_batch
        self.max_batch_tokens = max_batch_tokens
        self.stats: dict[str, Any] = {}
        self._tokens: dict[str, list[int]] = {}
        self._pad = self.tokenizer.pad_token_id
        if self._pad is None:
            self._pad = self.tokenizer.eos_token_id
        self._prefix: tuple[tuple[int, ...], list[tuple[torch.Tensor, torch.Tensor]]] | None = None

    # -- setup ----------------------------------------------------------------

    def _slot_tokens(self, symbols: str) -> list[int]:
        """Token id of each slot symbol, checked to be one exact round-trip token."""
        if symbols in self._tokens:
            return self._tokens[symbols]
        ids = []
        for symbol in symbols:
            encoded = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(encoded) != 1 or self.tokenizer.decode(encoded) != symbol:
                raise ValueError(f"{self.name} does not tokenise slot {symbol!r} as one token")
            ids.append(encoded[0])
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.name} maps two of the slots {symbols!r} to one token")
        self._tokens[symbols] = ids
        return ids

    def _encode(
        self,
        state: str,
        question: Question,
        layout: Sequence[int] | None = None,
        slots: str = "",
    ) -> list[int]:
        turns = messages(state, question, layout, slots)
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.tokenizer.apply_chat_template(turns, enable_thinking=False, **kwargs)
        except TypeError:  # template does not know about thinking mode
            text = self.tokenizer.apply_chat_template(turns, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    # -- the trick ------------------------------------------------------------

    def _prefill(self, prefix: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run the shared prefix once and keep its keys/values, reused across calls."""
        key = tuple(prefix)
        if self._prefix is not None and self._prefix[0] == key:
            return self._prefix[1]
        ids = torch.tensor([prefix], device=self.device)
        with torch.inference_mode():
            out = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=True,
                logits_to_keep=1,
            )
        cache = out.past_key_values
        kv = [tuple(t.detach() for t in _layer_kv(cache, i)) for i in range(_cache_depth(cache))]
        self._prefix = (key, kv)
        return kv

    def _branch(self, kv: list[tuple[torch.Tensor, torch.Tensor]], width: int) -> DynamicCache:
        """A fresh cache whose rows all view the same prefill.

        `expand` costs nothing: the suffix forward concatenates onto these
        tensors rather than writing into them, so one prefill really does serve
        the whole batch.
        """
        cache = DynamicCache()
        for index, (keys, values) in enumerate(kv):
            cache.update(keys.expand(width, -1, -1, -1), values.expand(width, -1, -1, -1), index)
        return cache

    def _score_batch(self, kv, prefix_len: int, suffixes: list[list[int]]) -> torch.Tensor:
        """Next-token logits at the end of every suffix, in one forward pass.

        Suffixes are left-padded so that every row's real last token lands at
        index -1; the padding is masked out and given no position, so each row
        sees exactly prefix + its own tokens.
        """
        width = max(len(s) for s in suffixes)
        ids, mask, positions = [], [], []
        for suffix in suffixes:
            pad = width - len(suffix)
            ids.append([self._pad] * pad + suffix)
            mask.append([1] * prefix_len + [0] * pad + [1] * len(suffix))
            positions.append([0] * pad + list(range(prefix_len, prefix_len + len(suffix))))
        batch = {
            "input_ids": torch.tensor(ids, device=self.device),
            "attention_mask": torch.tensor(mask, device=self.device),
            "position_ids": torch.tensor(positions, device=self.device),
        }
        with torch.inference_mode():
            out = self.model(
                **batch,
                past_key_values=self._branch(kv, len(suffixes)),
                use_cache=True,
                logits_to_keep=1,
            )
        return out.logits[:, -1, :].float()

    def _chunks(self, prompts: list[list[int]], prefix_len: int):
        """Split the rows into batches that fit, by count and by KV footprint.

        Every row in a batch carries its own copy of the prefix in the cache, so
        the memory a batch needs goes as `width x (prefix + suffix)`. A short
        state can take the full `max_batch`; a nine-thousand-token document
        cannot, and would otherwise ask for more KV than the card has. The
        budget keeps the same code working across both instead of making the
        caller guess a batch size per document.
        """
        start = 0
        while start < len(prompts):
            width, widest = 0, 0
            while start + width < len(prompts) and width < self.max_batch:
                longest = max(widest, len(prompts[start + width]) - prefix_len)
                if width and (width + 1) * (prefix_len + longest) > self.max_batch_tokens:
                    break
                widest, width = longest, width + 1
            width = max(width, 1)
            yield prompts[start : start + width]
            start += width

    # -- public api -----------------------------------------------------------

    def ask(
        self,
        state: Any,
        questions: Iterable[Any],
        *,
        temperature: float | None = None,
        permutations: int | str | None = None,
        aggregate: str | None = None,
        slots: str | None = None,
        shuffle_slots: bool | None = None,
        min_probability: float | None = None,
        min_mass: float | None = None,
    ) -> list[Answer]:
        """Answer every question about one state. Returns one `Answer` per question.

        `permutations` scores each question under that many cyclic relabellings
        of its options and averages the results, which cancels the model's
        preference for particular answer slots. It costs one extra suffix per
        extra layout — the state is still prefilled once — so it is cheap, but
        it is off by default because it is not free. `"all"` uses a full cover,
        the only setting that removes an additive slot bias exactly.

        `slots` chooses the symbols the options wear: a preset name from
        `jeff.slots.SLOT_SETS` or a literal string. `shuffle_slots` draws each
        question its own subset instead of always using the first few, which
        stops one workload-wide favourite from forming. The draw is seeded by
        the question, so answers stay reproducible, and it happens once per
        question rather than once per layout — so the cyclic cover still sees a
        fixed symbol set and still cancels exactly.

        `min_probability` and `min_mass` are floors below which the answer is
        withheld: `Answer.value` becomes `None` and `Answer.abstained` is set,
        while `Answer.choice` still reports what the model would have said. The
        two catch different failures — a low probability means the model could
        not choose between your options, a low mass means it did not want to
        answer from your options at all.
        """
        started = time.perf_counter()
        temperature = self.temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        permutations = self.permutations if permutations is None else permutations
        aggregate = self.aggregate if aggregate is None else aggregate
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}")
        pool_symbols = resolve_slots(self.slots if slots is None else slots)
        shuffle = self.shuffle_slots if shuffle_slots is None else shuffle_slots
        min_probability = self.min_probability if min_probability is None else min_probability
        min_mass = self.min_mass if min_mass is None else min_mass
        if not 0.0 <= min_probability <= 1.0 or not 0.0 <= min_mass <= 1.0:
            raise ValueError("min_probability and min_mass are probabilities, in 0..1")
        questions = coerce_all(questions)
        text = render_state(state)

        # Each question keeps one symbol set across all of its layouts, so the
        # cover below still sends every option past every symbol exactly once.
        chosen = [
            take_slots(
                pool_symbols,
                len(question.options),
                key=question.key if shuffle else None,
                seed=self.seed,
            )
            for question in questions
        ]
        for symbols in set(chosen):
            self._slot_tokens(symbols)

        # One row per (question, option layout). They all share the same state,
        # so the extra layouts ride along on the same prefill.
        rows = [
            (index, layout)
            for index, question in enumerate(questions)
            for layout in cyclic_layouts(len(question.options), permutations)
        ]
        prompts = [
            self._encode(text, questions[index], layout, chosen[index]) for index, layout in rows
        ]

        prefix_len = _common_prefix_len(prompts)
        prefix = prompts[0][:prefix_len]
        kv = self._prefill(prefix) if prefix_len else []
        prefilled = time.perf_counter()

        # Rows stay in question order on purpose. A question's layouts all have
        # the same length, so the natural order is already length-clustered;
        # sorting globally breaks that up and measures slower. Repacking to
        # minimise padding is worse still — the optimum of that objective is a
        # batch size of one, which is five times slower here. Batch width is
        # what buys throughput, and padding is second-order.
        scored = []
        for chunk in self._chunks(prompts, prefix_len):
            if prefix_len:
                scored.append(self._score_batch(kv, prefix_len, [p[prefix_len:] for p in chunk]))
            else:
                scored.append(self._score_batch([], 0, chunk))
        logits = torch.cat(scored)

        # Undo each relabelling, then fold the layouts of one question together.
        # Normalising over the whole vocabulary first costs nothing we do not
        # already have, and it is the only way to see how much of the model's
        # attention the menu actually captured.
        samples: list[list[torch.Tensor]] = [[] for _ in questions]
        layouts: list[list[tuple[int, ...]]] = [[] for _ in questions]
        masses: list[list[float]] = [[] for _ in questions]
        for (index, layout), row in zip(rows, logits):
            # Normalise against the whole vocabulary without materialising it.
            shown = row[self._slot_tokens(chosen[index])] - torch.logsumexp(row, dim=-1)
            masses[index].append(float(shown.exp().sum()))
            layouts[index].append(layout)
            canonical = torch.empty_like(shown)
            canonical[list(layout)] = shown
            samples[index].append(torch.log_softmax(canonical, dim=-1))

        answers = []
        for index, (question, drawn) in enumerate(zip(questions, samples)):
            stack = torch.stack(drawn)
            pooled = pool(stack, aggregate)
            probabilities = torch.softmax(pooled / temperature, dim=-1)
            mass = sum(masses[index]) / len(masses[index])
            answers.append(
                Answer(
                    question,
                    tuple(probabilities.tolist()),
                    tuple(pooled.tolist()),
                    tuple(tuple(row.exp().tolist()) for row in stack),
                    tuple(layouts[index]),
                    chosen[index],
                    tuple(masses[index]),
                    abstained=float(probabilities.max()) < min_probability or mass < min_mass,
                )
            )
        self.stats = {
            "questions": len(questions),
            "forward_rows": len(rows),
            "prefix_tokens": prefix_len,
            "suffix_tokens": sum(len(p) - prefix_len for p in prompts),
            "saved_tokens": prefix_len * (len(prompts) - 1),
            "prefill_seconds": prefilled - started,
            "total_seconds": time.perf_counter() - started,
            "output_tokens": 0,
            "abstained": sum(answer.abstained for answer in answers),
        }
        return answers

    def decide(self, state: Any, question: Any, **kwargs: Any) -> Answer:
        """Answer a single question."""
        return self.ask(state, [question], **kwargs)[0]
