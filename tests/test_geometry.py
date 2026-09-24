import numpy as np
import pytest

from assistlabel.core.geometry import (
    box_iou,
    mask_to_polygons,
    nms_boxes,
    polygon_area,
    polygons_to_mask,
)


def test_box_iou_perfect_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert box_iou(a, a)[0, 0] == 1.0
    b = np.array([[100, 100, 110, 110]], dtype=float)
    assert box_iou(a, b)[0, 0] == 0.0


def test_box_iou_partial():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 0, 15, 10]], dtype=float)
    # intersection 5x10=50, union 100+100-50=150
    assert abs(box_iou(a, b)[0, 0] - 50 / 150) < 1e-9


def test_nms_suppresses_overlap_keeps_distinct():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms_boxes(boxes, scores, iou_thres=0.5)
    assert keep == [0, 2]


def test_mask_to_polygons_roundtrip():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:70] = True
    polys = mask_to_polygons(mask, epsilon_ratio=0.001)
    assert len(polys) == 1
    rebuilt = polygons_to_mask(polys, 100, 100)
    iou = (mask & rebuilt).sum() / max((mask | rebuilt).sum(), 1)
    assert iou > 0.95


def test_mask_to_polygons_drops_tiny_components():
    mask = np.zeros((200, 200), dtype=bool)
    mask[10:100, 10:100] = True          # big
    mask[150:155, 150:155] = True        # tiny speck (25 px vs 8100 px)
    polys = mask_to_polygons(mask, min_component_area_ratio=0.2)
    assert len(polys) == 1
    rebuilt = polygons_to_mask(polys, 200, 200)
    assert rebuilt[152, 152] is False or not rebuilt[152, 152]


def test_mask_to_polygons_empty():
    assert mask_to_polygons(np.zeros((50, 50), dtype=bool)) == []


def test_polygon_area():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert polygon_area(square) == 100.0
