"""Depth Anything 3 engine (ByteDance-Seed/Depth-Anything-3).

Single transformer for any-view depth (+ pose). For our single-image batch
ground truth the Apache-2.0 variants are used:

- ``DA3METRIC-LARGE``: metric scale (via the model's estimated focal);
- ``DA3MONO-LARGE``: relative depth, predicted directly (not disparity).

Requires the DA3 package installed from source (not on PyPI)::

    pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3"
    pip install einops omegaconf imageio huggingface_hub

Weights (config.json + model.safetensors, PyTorchModelHubMixin layout) are
resolved ModelScope-first like every other engine.
"""

from __future__ import annotations

import numpy as np

from ..core.engine import BaseEngine, EngineError

INSTALL_HINT = (
    "DA3 engine needs the official package:\n"
    '  pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3"\n'
    "  pip install einops omegaconf imageio"
)

REQUIRED_FILES = ["config.json", "model.safetensors"]


class DepthAnything3Engine(BaseEngine):
    name = "depth_anything3"

    def __init__(self, hf_id: str, params: dict | None = None):
        self.hf_id = hf_id
        self.params = params or {}
        self.device = "cpu"
        self.resolved_dir: str | None = None
        self._model = None
        self._torch = None

    def load(self, device: str = "cuda", half: bool = False, source: str = "auto") -> None:
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise EngineError(INSTALL_HINT) from e

        self._torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

        from .. import hub

        self.resolved_dir = hub.resolve_model_dir_files(
            self.hf_id, self.params.get("ms_id", ""), REQUIRED_FILES, source=source
        )
        load_from = self.resolved_dir or self.hf_id
        offline = self.resolved_dir is not None

        self._model = DepthAnything3.from_pretrained(
            load_from, local_files_only=offline
        )
        self._model = self._model.to(self.device)
        self._model.eval()

    def infer(self, image_rgb: np.ndarray, **params) -> dict:
        return self.infer_batch([image_rgb], **params)[0]

    def infer_batch(self, images_rgb: list[np.ndarray], **params) -> list[dict]:
        """True batched inference: DA3's native API accepts an image list."""
        if self._model is None:
            raise EngineError(f"{self.name}: call load() before infer()")
        prediction = self._model.inference(list(images_rgb))
        outs: list[dict] = []
        for im, depth in zip(images_rgb, prediction.depth):
            d = np.asarray(depth, dtype=np.float32)
            # DA3 returns depth at the processing resolution (process_res, default
            # 504); our pipeline contract is full original size (detections and
            # depth must share one coordinate frame).
            h, w = im.shape[:2]
            if d.shape != (h, w):
                import cv2

                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            outs.append({
                "depth_m": d,
                "meta": {
                    "model": self.hf_id,
                    "dtype": "fp32(da3-autocast-bf16)",
                },
            })
        return outs

    def unload(self) -> None:
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
