"""Fuse depth maps with detections: per-object depth statistics.

Median inside the mask is the headline stat (robust to mask-edge noise and
to the occasional mismatched pixel at object borders). Invalid depth pixels
(zero / NaN) are excluded from all statistics.
"""

from __future__ import annotations

import numpy as np

from ..core.engine import Detection


def mask_depth_stats(
    depth_m: np.ndarray, mask: np.ndarray | None = None, box=None
) -> dict:
    """Depth statistics for one object region.

    ``mask`` (bool HxW) wins when given; otherwise the interior of ``box``
    (xyxy) is used. Returns {} when the region has no valid depth.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    if mask is not None:
        region = (np.asarray(mask) > 0) & valid
    elif box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, depth.shape[1]), min(y2, depth.shape[0])
        if x2 <= x1 or y2 <= y1:
            return {}
        region = np.zeros_like(valid)
        region[y1:y2, x1:x2] = True
        region &= valid
    else:
        raise ValueError("mask_depth_stats needs a mask or a box")

    values = depth[region]
    if values.size == 0:
        return {}

    return {
        "depth_median_m": round(float(np.median(values)), 3),
        "depth_min_m": round(float(values.min()), 3),
        "depth_max_m": round(float(values.max()), 3),
        "valid_ratio": round(float(values.size) / float(max(region.size, 1)), 4),
    }


# fuse.stats config uses short names; result keys are prefixed for clarity
STAT_KEYS = {
    "median": "depth_median_m",
    "min": "depth_min_m",
    "max": "depth_max_m",
}


def annotate_detections_with_depth(
    detections: list[Detection], depth_m: np.ndarray, stats: list[str] | None = None
) -> list[Detection]:
    """Fill ``det.extra`` with depth statistics in place. ``stats`` accepts
    short names (``median``/``min``/``max``, as used in run.yaml) or the full
    result keys; ``valid_ratio`` is always kept."""
    keep = {STAT_KEYS.get(s, s) for s in stats} if stats else None
    for det in detections:
        region_mask = det.mask if det.mask is not None else None
        result = mask_depth_stats(depth_m, mask=region_mask, box=None if det.mask is not None else det.box)
        if not result:
            continue
        if keep is not None:
            result = {k: v for k, v in result.items() if k in keep or k == "valid_ratio"}
        det.extra.update(result)
    return detections
