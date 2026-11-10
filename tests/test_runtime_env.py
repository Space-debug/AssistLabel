"""Runtime environment defaults applied before torch initializes CUDA."""

import os

from assistlabel.runner import quiet_progress


def test_quiet_progress_enables_expandable_segments(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    quiet_progress()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_quiet_progress_respects_user_override(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    quiet_progress()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
