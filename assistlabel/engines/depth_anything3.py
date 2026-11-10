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

# DA3's InputProcessor defaults (api.inference: process_res=504,
# process_res_method="upper_bound_resize"); used to predict the processed
# batch shape before inference.
PROCESS_RES = 504
PATCH_SIZE = 14


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

        # Weights stay fp32 on purpose: DA3's depth/pose heads run inside
        # ``torch.autocast(enabled=False)`` while feeding fp32 inputs, so bf16
        # weights would raise dtype mismatches. Compute is already bf16 autocast
        # (the package picks bf16/fp16 itself), so ``half`` has no extra effect.
        self._model = DepthAnything3.from_pretrained(
            load_from, local_files_only=offline
        )
        self._model = self._model.to(self.device)
        self._model.eval()
        self._use_inline_preprocess()

    def _use_inline_preprocess(self) -> None:
        """Preprocess batches inline instead of DA3's per-batch 8-thread pool.

        The pipeline already decodes images on a producer thread; letting DA3
        spawn a fresh ThreadPool for every batch adds overhead and competes for
        the GIL. Falls back silently if the upstream API changes.
        """
        try:
            from depth_anything_3.utils.io.input_processor import InputProcessor
        except Exception:  # noqa: BLE001 - optional/private upstream API
            return

        class _InlineInputProcessor(InputProcessor):
            def __call__(self, *args, **kwargs):
                kwargs["num_workers"] = 1
                kwargs["sequential"] = True
                return super().__call__(*args, **kwargs)

        self._model.input_processor = _InlineInputProcessor()

    def batch_key(self, image_rgb: np.ndarray) -> tuple[int, int]:
        """Processed (H, W) this image will have after DA3's input processor.

        DA3 center-crops a batch to the smallest member when processed sizes
        differ (``InputProcessor._unify_batch_shapes``), so only images with the
        same processed shape may share a batch — otherwise large images lose
        their borders and the depth is stretched back over the wrong geometry.
        """
        h, w = image_rgb.shape[:2]
        scale = PROCESS_RES / float(max(w, h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        def nearest_multiple(x: int) -> int:
            down = (x // PATCH_SIZE) * PATCH_SIZE
            up = down + PATCH_SIZE
            return up if abs(up - x) <= abs(x - down) else down

        return nearest_multiple(nh), nearest_multiple(nw)

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
                    "dtype": "fp32-weights/bf16-autocast",
                },
            })
        return outs

    def unload(self) -> None:
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
