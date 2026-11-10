"""Inference engine abstraction: lazy loading, unload for GPU sharing.

All engines take/return **numpy** arrays: RGB uint8 images in, plain dicts /
dataclasses out. Torch / SAM3 / transformers imports happen inside ``load()``
so the package stays importable (and testable) on machines without GPU deps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One instance result. Coordinates are xyxy floats in image space."""

    label: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray | None = None  # bool (H, W), image-sized
    extra: dict = field(default_factory=dict)


class EngineError(RuntimeError):
    """Raised when an engine's heavy dependencies are missing or fail."""


class BaseEngine(ABC):
    """Lazily-loaded inference engine; unload() frees GPU memory so a second
    engine can reuse the card (depth -> detect serial execution)."""

    name: str = "base"

    @abstractmethod
    def load(self, device: str = "cuda", half: bool = False, source: str = "auto") -> None: ...

    @abstractmethod
    def infer(self, image_rgb: np.ndarray, **params): ...

    def infer_batch(self, images_rgb: list[np.ndarray], **params) -> list:
        """Batch inference. Default: sequential loop; engines with true batch
        support (GPU) override this."""
        return [self.infer(img, **params) for img in images_rgb]

    def batch_key(self, image_rgb: np.ndarray) -> tuple[int, int]:
        """Grouping key for batch inference.

        The pipeline only batches consecutive images with the same key.
        Engines that resize/pad a batch to a common shape must override this
        so images that would be cropped together never share a batch.
        """
        return image_rgb.shape[:2]

    def unload(self) -> None:  # pragma: no cover - trivial
        pass


def require(import_error: ImportError, package: str, hint: str) -> EngineError:
    return EngineError(
        f"Missing dependency '{package}' ({import_error}). {hint}"
    )


def weight_dtype(torch, device: str, half: bool):
    """Model weight dtype for the requested precision.

    ``half=True`` means **bf16** on CUDA, not fp16: bf16 has the same exponent
    range as fp32, so depth / box regression cannot overflow or underflow.
    Compute kernels already run under bf16 autocast in both engines; this only
    halves the resident weight footprint (~2.4 GB saved for DA3-Metric-L+SAM3).
    """
    if half and device == "cuda":
        return torch.bfloat16
    return torch.float32
