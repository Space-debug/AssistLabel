"""Overlay rendering: detection masks/boxes on images, depth pseudo-color
side-by-side. Pure OpenCV; used for the ``viz/`` QA images."""

from __future__ import annotations

import cv2
import numpy as np

from ..core.engine import Detection
from ..io.depth_io import pseudocolor

# 20 distinguishable BGR colors, cycled by label hash
PALETTE = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (0, 0, 0),
    (100, 100, 255), (255, 100, 100), (100, 255, 100), (180, 120, 60),
    (60, 180, 180), (150, 60, 180), (180, 180, 60), (60, 60, 180),
    (120, 255, 180), (255, 120, 255), (180, 255, 120), (90, 90, 90),
]

# per-class palette shared with semantic_viz (same hue for the same class,
# index = class id). Kept in BGR order for cv2.
CLASS_PALETTE = np.array([
    [0, 0, 0], [180, 119, 31], [14, 127, 255], [44, 160, 44],
    [40, 39, 214], [189, 103, 148], [75, 86, 140], [194, 119, 227],
], dtype=np.uint8)


def _color_for(label: str) -> tuple[int, int, int]:
    idx = int.from_bytes(label.encode("utf-8")[:4], "little") % len(PALETTE)
    return PALETTE[idx]


def _color_for_class(label: str, class_ids: dict[str, int] | None):
    """BGR color for a label; consistent with semantic_viz when class_ids given."""
    if class_ids and label in class_ids:
        cid = class_ids[label]
        b, g, r = [int(v) for v in CLASS_PALETTE[cid % len(CLASS_PALETTE)]]
        return (r, g, b)  # palette stored RGB for readability; cv2 wants BGR
    return _color_for(label)


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[Detection],
    draw_masks: bool = True,
    alpha: float = 0.4,
    with_depth: bool = True,
    class_ids: dict[str, int] | None = None,
) -> np.ndarray:
    """Render detections: translucent masks + boxes + label/score/(depth).

    With ``class_ids`` (label -> class id) the colors match semantic_viz
    per class.
    """
    canvas = image_bgr.copy()
    if draw_masks:
        overlay = canvas.copy()
        for det in detections:
            if det.mask is not None:
                overlay[det.mask > 0] = _color_for_class(det.label, class_ids)
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    for det in detections:
        color = _color_for_class(det.label, class_ids)
        x1, y1, x2, y2 = [int(round(v)) for v in det.box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label} {det.score:.2f}"
        if with_depth and "depth_median_m" in det.extra:
            text += f" {det.extra['depth_median_m']:.1f}m"
        ty = y1 - 6 if y1 - 6 > 12 else y2 + 18
        # 深色底衬 + 单次文字绘制（描边双绘会视觉上变成两个标签）
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            canvas,
            (max(x1 - 1, 0), max(ty - th - 3, 0)),
            (min(x1 + tw + 3, canvas.shape[1] - 1), min(ty + 2, canvas.shape[0] - 1)),
            (30, 30, 30),
            -1,
        )
        cv2.putText(
            canvas, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    return canvas


def depth_side_by_side(image_bgr: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Left: image, right: pseudo-color depth. Heights matched."""
    colored = pseudocolor(depth_m)
    colored = cv2.resize(colored, (image_bgr.shape[1], image_bgr.shape[0]))
    return np.hstack([image_bgr, colored])
