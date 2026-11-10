"""Optional GPU precision regression (not part of the default suite).

Run explicitly with real weights and real images::

    ASSISTLABEL_GPU_TESTS=1 \
    ASSISTLABEL_GPU_IMAGES=D:/data/some_images \
    python -m pytest tests/test_precision_gpu.py -q -s

Checks:
- SAM3 bf16 weights stay numerically close to fp32 (count/box/mask/score).
- Decoder-side prompt batching matches one-prompt-per-forward.
- DA3 batched inference matches single-image inference (bucket safety).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

GPU = os.environ.get("ASSISTLABEL_GPU_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not GPU, reason="set ASSISTLABEL_GPU_TESTS=1 to run GPU precision tests"
)

IMAGES_ENV = "ASSISTLABEL_GPU_IMAGES"
PROMPTS = os.environ.get("ASSISTLABEL_GPU_PROMPTS", "chair,dining table,plastic water bottle,cabinet")
N_IMAGES = int(os.environ.get("ASSISTLABEL_GPU_N", "5"))


def _require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch


def _images(n=N_IMAGES):
    root = os.environ.get(IMAGES_ENV)
    if not root:
        pytest.skip(f"set {IMAGES_ENV} to a directory of test images")
    paths = sorted(
        p for p in Path(root).rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )[:n]
    if not paths:
        pytest.skip(f"no images under {root}")
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not out:
        pytest.skip(f"no decodable images under {root}")
    return out


def _match(ref, new):
    """Match detections by label + best box IoU; returns (box_ious, mask_ious, score_diff, n)."""
    from assistlabel.core.geometry import box_iou

    used = set()
    box_ious, mask_ious, score_diff = [], [], 0.0
    for d in new:
        best, best_iou = None, 0.0
        for k, r in enumerate(ref):
            if k in used or r.label != d.label:
                continue
            iou = float(box_iou(np.asarray(d.box)[None], np.asarray(r.box)[None])[0, 0])
            if iou > best_iou:
                best, best_iou = k, iou
        if best is None or best_iou < 0.5:
            continue
        used.add(best)
        box_ious.append(best_iou)
        score_diff = max(score_diff, abs(d.score - ref[best].score))
        if d.mask is not None and ref[best].mask is not None:
            mask_ious.append(
                (d.mask & ref[best].mask).sum() / max((d.mask | ref[best].mask).sum(), 1)
            )
    return box_ious, mask_ious, score_diff, len(used)


def _sam3_run(images, half, prompt_batch):
    from assistlabel.engines.sam3 import Sam3Engine

    prompts = [p.strip() for p in PROMPTS.split(",") if p.strip()]
    engine = Sam3Engine(params={"prompt_batch": prompt_batch})
    engine.load(device="cuda", half=half)
    try:
        return [
            engine.infer(im, prompts=prompts, conf_thres=0.4, max_objects_per_prompt=50)
            for im in images
        ]
    finally:
        engine.unload()


def test_sam3_bf16_matches_fp32():
    _require_cuda()
    images = _images()
    ref = _sam3_run(images, half=False, prompt_batch=4)
    new = _sam3_run(images, half=True, prompt_batch=4)

    total_ref = sum(len(r) for r in ref)
    total_new = sum(len(n) for n in new)
    assert abs(total_new - total_ref) <= max(2, int(0.05 * total_ref)), (total_ref, total_new)

    boxes, masks, score_diff, matched = _match(
        [d for r in ref for d in r], [d for n in new for d in n]
    )
    if total_ref:
        assert matched >= 0.95 * total_ref
        assert min(boxes) >= 0.98
        if masks:
            assert min(masks) >= 0.98
        assert score_diff <= 0.02


def test_sam3_prompt_batching_matches_sequential():
    _require_cuda()
    images = _images()
    ref = _sam3_run(images, half=True, prompt_batch=1)
    new = _sam3_run(images, half=True, prompt_batch=4)

    total_ref = sum(len(r) for r in ref)
    total_new = sum(len(n) for n in new)
    assert abs(total_new - total_ref) <= max(2, int(0.05 * total_ref))

    boxes, masks, score_diff, matched = _match(
        [d for r in ref for d in r], [d for n in new for d in n]
    )
    if total_ref:
        assert matched >= 0.95 * total_ref
        assert min(boxes) >= 0.98
        if masks:
            assert min(masks) >= 0.98


def test_da3_batch_matches_single_image():
    torch = _require_cuda()
    from assistlabel.engines.depth_anything3 import DepthAnything3Engine

    images = _images(n=2)
    if len(images) < 2:
        pytest.skip("need at least two images")
    h = min(im.shape[0] for im in images)
    w = min(im.shape[1] for im in images)
    pair = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in images[:2]]

    engine = DepthAnything3Engine("depth-anything/DA3METRIC-LARGE")
    engine.load(device="cuda")
    try:
        single = engine.infer(pair[0])["depth_m"]
        batch = engine.infer_batch(pair)[0]["depth_m"]
    finally:
        engine.unload()

    valid = (single > 0) & np.isfinite(single)
    rel = np.abs(batch[valid] - single[valid]) / np.maximum(single[valid], 1e-6)
    assert float(np.median(rel)) < 1e-3
    assert float(np.percentile(rel, 99)) < 1e-2
