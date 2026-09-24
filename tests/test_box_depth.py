import numpy as np
import pytest

from assistlabel.core.engine import Detection
from assistlabel.fusion.box_depth import annotate_detections_with_depth, mask_depth_stats


def test_mask_stats_ignores_invalid():
    depth = np.full((20, 20), 5.0, dtype=np.float32)
    depth[15, 15] = 500.0       # outlier inside mask
    clean = np.zeros((20, 20), dtype=bool)
    clean[0:10, 10:20] = True   # fully valid region: all 5.0
    outlier = np.zeros((20, 20), dtype=bool)
    outlier[12:18, 12:18] = True  # region containing the outlier

    stats = mask_depth_stats(depth, mask=clean)
    assert stats["depth_median_m"] == 5.0
    assert stats["depth_min_m"] == 5.0

    stats2 = mask_depth_stats(depth, mask=outlier)
    assert stats2["depth_median_m"] == 5.0      # median robust to outlier
    assert stats2["depth_max_m"] == 500.0
    assert 0 < stats2["valid_ratio"] <= 1.0


def test_mask_stats_all_invalid_returns_empty():
    depth = np.zeros((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=bool)
    assert mask_depth_stats(depth, mask=mask) == {}


def test_box_stats():
    depth = np.full((30, 30), 7.25, dtype=np.float32)
    stats = mask_depth_stats(depth, box=(5, 5, 15, 15))
    assert stats["depth_median_m"] == 7.25
    with pytest.raises(ValueError):
        mask_depth_stats(depth)


def test_annotate_detections_writes_extra():
    depth = np.full((50, 50), 3.0, dtype=np.float32)
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True
    det = Detection(label="car", score=0.9, box=(10, 10, 30, 30), mask=mask)
    annotate_detections_with_depth([det], depth)
    assert det.extra["depth_median_m"] == 3.0
    assert "depth_min_m" in det.extra and "depth_max_m" in det.extra


def test_annotate_box_only_detection():
    depth = np.full((50, 50), 4.0, dtype=np.float32)
    det = Detection(label="car", score=0.9, box=(10, 10, 30, 30), mask=None)
    annotate_detections_with_depth([det], depth)
    assert det.extra["depth_median_m"] == 4.0
