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

from .prompt import Answer, LETTERS, Question, coerce_all, messages, render_state

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


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
        temperature: float = 1.0,
        max_batch: int = 16,
    ) -> None:
        self.device = _pick_device(device)
        if dtype is None:
            dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=dtype, attn_implementation="sdpa"
        ).to(self.device).eval()
        self.name = model
        self.temperature = temperature
        self.max_batch = max_batch
        self.stats: dict[str, Any] = {}
        self._slots = self._answer_slots()
        self._pad = self.tokenizer.pad_token_id
        if self._pad is None:
            self._pad = self.tokenizer.eos_token_id
        self._prefix: tuple[tuple[int, ...], list[tuple[torch.Tensor, torch.Tensor]]] | None = None

    # -- setup ----------------------------------------------------------------

    def _answer_slots(self) -> list[int]:
        """Token id of each answer letter, checked to be one exact round-trip token."""
        slots = []
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids) != letter:
                raise ValueError(f"{self.name} does not tokenise answer slot {letter!r} as one token")
            slots.append(ids[0])
        if len(set(slots)) != len(slots):
            raise ValueError(f"{self.name} maps two answer slots to the same token")
        return slots

    def _encode(self, state: str, question: Question) -> list[int]:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.tokenizer.apply_chat_template(
                messages(state, question), enable_thinking=False, **kwargs
            )
        except TypeError:  # template does not know about thinking mode
            text = self.tokenizer.apply_chat_template(messages(state, question), **kwargs)
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

    # -- public api -----------------------------------------------------------

    def ask(
        self,
        state: Any,
        questions: Iterable[Any],
        *,
        temperature: float | None = None,
    ) -> list[Answer]:
        """Answer every question about one state. Returns one `Answer` per question."""
        started = time.perf_counter()
        temperature = self.temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        questions = coerce_all(questions)
        text = render_state(state)
        prompts = [self._encode(text, q) for q in questions]

        prefix_len = _common_prefix_len(prompts)
        prefix = prompts[0][:prefix_len]
        kv = self._prefill(prefix) if prefix_len else []
        prefilled = time.perf_counter()

        rows = []
        for start in range(0, len(prompts), self.max_batch):
            chunk = prompts[start : start + self.max_batch]
            if prefix_len:
                rows.append(self._score_batch(kv, prefix_len, [p[prefix_len:] for p in chunk]))
            else:
                rows.append(self._score_batch([], 0, chunk))
        logits = torch.cat(rows)

        answers = []
        for question, row in zip(questions, logits):
            slots = row[self._slots[: len(question.options)]]
            probabilities = torch.softmax(slots / temperature, dim=-1)
            answers.append(
                Answer(question, tuple(probabilities.tolist()), tuple(slots.tolist()))
            )
        self.stats = {
            "questions": len(questions),
            "prefix_tokens": prefix_len,
            "suffix_tokens": sum(len(p) - prefix_len for p in prompts),
            "saved_tokens": prefix_len * (len(prompts) - 1),
            "prefill_seconds": prefilled - started,
            "total_seconds": time.perf_counter() - started,
            "output_tokens": 0,
        }
        return answers

    def decide(self, state: Any, question: Any, **kwargs: Any) -> Answer:
        """Answer a single question."""
        return self.ask(state, [question], **kwargs)[0]
