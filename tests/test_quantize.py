"""The 4-bit switch refuses what it cannot do before touching the weights."""

from __future__ import annotations

import pytest
import torch

from jeff.engine import _load_model


def test_rejects_unknown_quantisation():
    with pytest.raises(ValueError, match="quantize"):
        _load_model("unused", torch.device("cpu"), torch.float32, "3bit")


def test_4bit_needs_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        _load_model("unused", torch.device("cpu"), torch.float32, "4bit")
