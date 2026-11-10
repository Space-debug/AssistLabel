"""Runtime environment defaults applied before torch initializes CUDA."""

import os
import types

from assistlabel.core.engine import weight_dtype
from assistlabel.runner import quiet_progress


def test_quiet_progress_enables_expandable_segments(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    quiet_progress()
    if os.name == "nt":
        # the Windows CUDA allocator does not support it; don't provoke a warning
        assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    else:
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_quiet_progress_respects_user_override(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    quiet_progress()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


def test_weight_dtype_uses_bf16_on_cuda_only():
    fake_torch = types.SimpleNamespace(bfloat16="bf16", float32="fp32")
    assert weight_dtype(fake_torch, "cuda", True) == "bf16"
    assert weight_dtype(fake_torch, "cuda", False) == "fp32"
    assert weight_dtype(fake_torch, "cpu", True) == "fp32"
