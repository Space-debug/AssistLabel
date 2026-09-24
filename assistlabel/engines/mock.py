"""Mock engines: deterministic, dependency-free results for pipeline tests
and ``--dry-run`` style verification of the whole flow without GPU deps.

MockDepth: smooth parabolic depth field (0.5 m .. 20 m).
MockDetect: one 'object' centered in the image + a corner box, labels
``mock_object`` / ``mock_corner`` with fixed scores.
"""

from __future__ import annotations

import numpy as np

from ..core.engine import BaseEngine, Detection


class MockDepthEngine(BaseEngine):
    name = "mock_depth"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def load(self, device: str = "cuda", half: bool = False, source: str = "auto") -> None:
        pass

    def infer(self, image_rgb: np.ndarray, **params) -> dict:
        h, w = image_rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        depth = 0.5 + 19.5 * ((xx / max(w - 1, 1)) * 0.6 + (yy / max(h - 1, 1)) * 0.4)
        return {"depth_m": depth.astype(np.float32), "meta": {"model": "mock"}}


class MockDetectEngine(BaseEngine):
    name = "mock_detect"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def load(self, device: str = "cuda", half: bool = False, source: str = "auto") -> None:
        pass

    def infer(self, image_rgb: np.ndarray, prompts: list[str] | None = None, **params):
        h, w = image_rgb.shape[:2]
        labels = prompts or ["mock_object"]
        dets: list[Detection] = []
        for label in labels:
            cx, cy = w * 0.5, h * 0.5
            bw, bh = w * 0.3, h * 0.3
            mask = np.zeros((h, w), dtype=bool)
            mask[int(cy - bh / 2) : int(cy + bh / 2), int(cx - bw / 2) : int(cx + bw / 2)] = True
            dets.append(
                Detection(
                    label=label,
                    score=0.9,
                    box=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                    mask=mask,
                )
            )
        return dets
