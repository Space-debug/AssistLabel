"""Pure numpy/cv2 geometry helpers: IoU, NMS, mask<->polygon conversion.

Kept free of torch so the whole module is unit-testable without GPU deps.
Coordinate convention: boxes are xyxy ``[x1, y1, x2, y2]`` float arrays of
shape ``(N, 4)``; masks are ``np.bool_`` arrays of shape ``(H, W)``.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "box_iou",
    "nms_boxes",
    "mask_to_polygons",
    "polygons_to_mask",
    "polygon_area",
]


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (N, M) float array."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.7) -> list[int]:
    """Greedy NMS on xyxy boxes; returns kept indices sorted by score desc.

    Cross-class suppression is intentionally *not* performed here — callers
    run NMS per class.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i][None, :], boxes[rest])[0]
        order = rest[ious <= iou_thres]
    return keep


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of an (N, 2) polygon."""
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def mask_to_polygons(
    mask: np.ndarray,
    epsilon_ratio: float = 0.002,
    min_component_area_ratio: float = 0.05,
    max_components: int = 4,
    max_points: int = 500,
) -> list[np.ndarray]:
    """Convert a boolean instance mask into simplified polygons (Nx2 int32).

    - Contours are extracted with RETR_EXTERNAL (holes dropped).
    - Each contour is simplified with Douglas-Peucker, ``epsilon =
      epsilon_ratio * perimeter``.
    - Disjoint components smaller than ``min_component_area_ratio`` of the
      largest component are dropped; at most ``max_components`` are returned.
    - If simplification leaves more than ``max_points`` vertices, the polygon
      is re-simplified with progressively larger epsilon.
    """
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    areas = [cv2.contourArea(c) for c in contours]
    largest = max(areas) if areas else 0.0
    if largest <= 0:
        return []

    picked = [
        (c, a)
        for c, a in zip(contours, areas)
        if a >= min_component_area_ratio * largest
    ]
    picked.sort(key=lambda ca: ca[1], reverse=True)
    picked = picked[:max_components]

    polygons: list[np.ndarray] = []
    for contour, _area in picked:
        poly = _simplify(contour, epsilon_ratio, max_points)
        if poly is not None and len(poly) >= 3:
            polygons.append(poly)
    return polygons


def _simplify(contour: np.ndarray, epsilon_ratio: float, max_points: int) -> np.ndarray | None:
    eps = epsilon_ratio * cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, eps, closed=True)
    tries = 0
    while len(approx) > max_points and tries < 6:
        eps *= 2.0
        approx = cv2.approxPolyDP(contour, eps, closed=True)
        tries += 1
    if approx is None or len(approx) < 3:
        return None
    return approx.reshape(-1, 2).astype(np.int32)


def polygons_to_mask(polygons: list[np.ndarray], width: int, height: int) -> np.ndarray:
    """Rasterize polygons (list of Nx2) into a single boolean mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        poly = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)
