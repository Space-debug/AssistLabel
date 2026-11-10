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

from ..core.engine import BaseEngine, Detection, EngineError, weight_dtype
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

        dtype = weight_dtype(torch, self.device, half)
        self._model = Sam3Model.from_pretrained(
            load_from,
            dtype=dtype,
            local_files_only=self.resolved_dir is not None,
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
                for label, dets in zip(
                    txt,
                    self._postprocess(
                        outputs, target_sizes, conf_thres,
                        max_objects_per_prompt, nms_iou,
                    ),
                ):
                    for det in dets:
                        det.label = label
                    out.append(dets)
        return out

    def _postprocess(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        conf_thres: float,
        max_objects: int,
        nms_iou: float,
    ) -> list[list[Detection]]:
        """Raw SAM3 outputs -> detections, without materializing full-resolution
        masks for every candidate.

        ``processor.post_process_instance_segmentation`` upsamples *all*
        above-threshold queries to the original image and casts them to int64
        (16.6 MB per 1080p mask). We instead threshold + NMS in normalized box
        space, keep at most ``max_objects``, and upsample only those survivors
        one at a time as bool. Peak memory drops by an order of magnitude while
        boxes/scores/masks stay numerically equivalent.
        """
        torch = self._torch
        scores_all = outputs.pred_logits.sigmoid()
        if getattr(outputs, "presence_logits", None) is not None:
            scores_all = scores_all * outputs.presence_logits.sigmoid()
        boxes_all = outputs.pred_boxes
        masks_all = outputs.pred_masks

        results: list[list[Detection]] = []
        for idx, (h, w) in enumerate(target_sizes):
            scores = scores_all[idx]
            keep = scores > conf_thres
            if not bool(keep.any()):
                results.append([])
                continue

            kept = torch.nonzero(keep, as_tuple=False).flatten()
            scale = torch.tensor([w, h, w, h], device=boxes_all.device, dtype=boxes_all.dtype)
            boxes = (boxes_all[idx][kept] * scale).detach().float().cpu().numpy()
            scores_np = scores[kept].detach().float().cpu().numpy()
            order = nms_boxes(boxes.astype(np.float64), scores_np, iou_thres=nms_iou)[:max_objects]

            detections: list[Detection] = []
            for pos in order:
                mask = None
                if masks_all is not None:
                    logit = masks_all[idx][kept[pos]]
                    # sigmoid *before* upsampling: interpolating logits and
                    # thresholding at 0 would shrink every mask at the boundary
                    prob = torch.sigmoid(logit.float())
                    up = torch.nn.functional.interpolate(
                        prob[None, None], size=(h, w),
                        mode="bilinear", align_corners=False,
                    )[0, 0]
                    mask = (up > 0.5).cpu().numpy()
                detections.append(
                    Detection(
                        label=None,  # filled by the caller
                        score=float(scores_np[pos]),
                        box=tuple(float(v) for v in boxes[pos]),
                        mask=mask,
                    )
                )
            results.append(detections)
        return results
