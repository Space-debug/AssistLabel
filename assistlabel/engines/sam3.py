"""SAM3 engine via HuggingFace transformers — pure pip, no repo clone.

transformers >= 5.x ships ``Sam3Model`` / ``Sam3Processor``. Weights are the
HF-format ``model.safetensors`` (``detector_model.*`` keys) on the ModelScope
mirror of ``facebook/sam3``. (Meta's native ``sam3.pt`` uses ``detector.*``
keys and is a different, incompatible format — this tool does not use it.)

ModelScope's mirror lacks ``preprocessor_config.json``, so the processor is
assembled from class defaults (``Sam3Processor(Sam3ImageProcessor(), tokenizer)``);
if the file is present (full HF cache) ``from_pretrained`` is preferred.
"""

from __future__ import annotations

import numpy as np

from ..core.engine import BaseEngine, Detection, EngineError
from ..core.geometry import nms_boxes

INSTALL_HINT = (
    "sam3 engine needs transformers>=5 with SAM3 support"
    " (pip install 'transformers>=5') and torchvision."
)

# Files required for offline transformers loading of facebook/sam3
REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "special_tokens_map.json",
]

DEFAULT_PROMPT_BATCH = 4  # prompts per forward; bounds VRAM (image repeated per prompt)


class Sam3Engine(BaseEngine):
    name = "sam3"

    def __init__(self, hf_id: str = "facebook/sam3", params: dict | None = None):
        self.hf_id = hf_id
        self.params = params or {}
        self.device = "cpu"
        self.resolved_dir: str | None = None
        self._model = None
        self._processor = None
        self._torch = None

    def load(self, device: str = "cuda", half: bool = False, source: str = "auto") -> None:
        try:
            import torch
            from transformers import (  # noqa: F401
                AutoTokenizer,
                Sam3ImageProcessor,
                Sam3Model,
                Sam3Processor,
            )
        except ImportError as e:
            raise EngineError(INSTALL_HINT) from e

        self._torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

        from .. import hub

        # targeted multi-file download (snapshot_download would also pull the
        # duplicate 3.4GB native sam3.pt from the mirror)
        self.resolved_dir = hub.resolve_model_dir_files(
            self.hf_id, self.params.get("ms_id", ""), REQUIRED_FILES, source=source
        )
        load_from = self.resolved_dir or self.hf_id  # None -> let transformers try HF hub

        self._model = Sam3Model.from_pretrained(
            load_from, local_files_only=self.resolved_dir is not None
        ).to(self.device).eval()
        try:
            self._processor = Sam3Processor.from_pretrained(
                load_from, local_files_only=self.resolved_dir is not None
            )
        except OSError:
            # mirror lacks preprocessor_config.json -> class defaults
            tokenizer = AutoTokenizer.from_pretrained(
                load_from, local_files_only=self.resolved_dir is not None
            )
            self._processor = Sam3Processor(Sam3ImageProcessor(), tokenizer)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # -- inference ----------------------------------------------------------
    @property
    def prompt_batch(self) -> int:
        """(image, prompt) pairs per forward pass; bounds VRAM."""
        return max(1, int(self.params.get("prompt_batch", DEFAULT_PROMPT_BATCH)))

    def infer(
        self,
        image_rgb: np.ndarray,
        prompts: list[str] | None = None,
        conf_thres: float = 0.5,
        max_objects_per_prompt: int = 50,
        nms_iou: float = 0.7,
        **params,
    ) -> list[Detection]:
        if not prompts:
            return []
        flat = self.infer_pairs(
            [image_rgb] * len(prompts), list(prompts),
            conf_thres=conf_thres, max_objects_per_prompt=max_objects_per_prompt,
            nms_iou=nms_iou, **params,
        )
        return [d for dets in flat for d in dets]

    def infer_pairs(
        self,
        images_rgb: list[np.ndarray],
        prompts: list[str],
        conf_thres: float = 0.5,
        max_objects_per_prompt: int = 50,
        nms_iou: float = 0.7,
        **params,
    ) -> list[list[Detection]]:
        """Batched (image, prompt) pairs -> one Detection list per pair.

        Pairs may span multiple images; the processor pairs ``images[i]`` with
        ``text[i]``. Chunked by ``prompt_batch`` to bound VRAM.
        """
        if self._model is None or self._processor is None:
            raise EngineError(f"{self.name}: call load() before infer()")
        if not prompts or len(prompts) != len(images_rgb):
            raise EngineError("infer_pairs needs prompts aligned with images")
        if len(images_rgb) == 0:
            return []

        torch = self._torch
        prompt_batch = self.prompt_batch
        out: list[list[Detection]] = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device == "cuda")):
            for i in range(0, len(images_rgb), prompt_batch):
                ims = list(images_rgb[i : i + prompt_batch])
                txt = list(prompts[i : i + prompt_batch])
                inputs = self._processor(images=ims, text=txt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                target_sizes = [(im.shape[0], im.shape[1]) for im in ims]
                segs = self._processor.post_process_instance_segmentation(
                    outputs, threshold=conf_thres, target_sizes=target_sizes
                )
                for label, res, size in zip(txt, segs, target_sizes):
                    out.append(self._filter(res, label, max_objects_per_prompt, nms_iou, size))
        return out

    def _filter(
        self,
        result: dict,
        label: str,
        max_objects: int,
        nms_iou: float,
        size: tuple[int, int],
    ) -> list[Detection]:
        # post-processed results come back on the model's device
        scores_t = result["scores"]
        boxes_t = result["boxes"]
        if hasattr(scores_t, "cpu"):
            scores_t = scores_t.detach().float().cpu()
            boxes_t = boxes_t.detach().float().cpu()
        scores = np.asarray(scores_t, dtype=np.float32).reshape(-1)
        boxes = np.asarray(boxes_t, dtype=np.float64).reshape(-1, 4)
        masks = result.get("masks")
        if len(scores) == 0:
            return []

        order = nms_boxes(boxes, scores, iou_thres=nms_iou)[:max_objects]
        detections: list[Detection] = []
        for i in order:
            mask = None
            if masks is not None and i < len(masks):
                m = masks[i]
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                else:
                    m = np.asarray(m)
                if m.shape == (size[0], size[1]):
                    mask = m.astype(bool)
            detections.append(
                Detection(
                    label=label,
                    score=float(scores[i]),
                    box=tuple(float(v) for v in boxes[i]),
                    mask=mask,
                )
            )
        return detections
