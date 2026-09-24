"""Depth Anything engine (transformers AutoModelForDepthEstimation).

Outputs float32 metric depth in **meters** at the original image size.
Relative-depth models output an inverse-depth-ish scale; the run config
should pick the right metric variant per domain (indoor/outdoor).
"""

from __future__ import annotations

import numpy as np

from ..core.engine import BaseEngine, EngineError
from ..io.depth_io import dequantize_uint16_to_m, quantize_m_to_uint16  # noqa: F401 (re-export convenience)

RELATIVE_NOTE = (
    "Relative-depth models have no absolute scale. For metric ground truth "
    "use a Metric-* variant (Indoor-Hypersim / Outdoor-VKITTI)."
)


class DepthAnythingEngine(BaseEngine):
    name = "depth_anything"

    def __init__(self, hf_id: str, params: dict | None = None):
        self.hf_id = hf_id
        self.params = params or {}
        self.device = "cpu"
        self._processor = None
        self._model = None
        self._torch = None

    def load(self, device: str = "cuda", half: bool = False, source: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise EngineError(
                "Depth engine needs torch + transformers. Install with:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
                "  pip install transformers"
            ) from e

        from ..hub import resolve_model_dir

        self._torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if (half and self.device == "cuda") else torch.float32

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True  # fixed-size inputs -> pick fastest kernels
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # weights: ModelScope first (source=auto), HuggingFace fallback
        self.resolved_dir = resolve_model_dir(
            self.hf_id, self.params.get("ms_id", ""), source=source
        )
        # local snapshot -> never touch the network; hf_id -> transformers downloads
        offline = self.resolved_dir != self.hf_id
        self._processor = AutoImageProcessor.from_pretrained(
            self.resolved_dir, local_files_only=offline
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.resolved_dir, torch_dtype=dtype, local_files_only=offline
        ).to(self.device)
        self._model.eval()

    def _forward(self, images_rgb: list[np.ndarray], sizes: list[tuple[int, int]]) -> list:
        """Shared processor->model->resize-to-original path."""
        torch = self._torch
        inputs = self._processor(images=images_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        use_autocast = self.device == "cuda" and next(self._model.parameters()).dtype == torch.float16
        with torch.no_grad():
            outputs = self._model(**inputs)
            pred = outputs.predicted_depth  # (B, h', w') or (h', w') for B=1
            if pred.ndim == 2:
                pred = pred.unsqueeze(0)
            out = []
            for i, (h, w) in enumerate(sizes):
                p = torch.nn.functional.interpolate(
                    pred[i:i + 1].unsqueeze(1).float(), size=(h, w),
                    mode="bicubic", align_corners=False,
                )
                out.append(p[0, 0].float().cpu().numpy())
        depth_meta = {"model": self.hf_id, "dtype": "fp16" if use_autocast else "fp32"}
        return [{"depth_m": d.astype(np.float32), "meta": depth_meta} for d in out]

    def infer(self, image_rgb: np.ndarray, **params) -> dict:
        if self._model is None:
            raise EngineError(f"{self.name}: call load() before infer()")
        h, w = image_rgb.shape[:2]
        return self._forward([image_rgb], [(h, w)])[0]

    def infer_batch(self, images_rgb: list[np.ndarray], **params) -> list:
        """True batched GPU inference. Falls back to per-image on any error
        (e.g. CUDA OOM, mixed-size processor quirks) so one bad image never
        poisons the batch."""
        if self._model is None:
            raise EngineError(f"{self.name}: call load() before infer()")
        if len(images_rgb) == 1 or self.device == "cpu":
            return [self.infer(im, **params) for im in images_rgb]
        try:
            sizes = [(im.shape[0], im.shape[1]) for im in images_rgb]
            return self._forward(list(images_rgb), sizes)
        except EngineError:
            raise
        except Exception:
            return [self.infer(im, **params) for im in images_rgb]

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
