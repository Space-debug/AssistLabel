"""mask_to_coco_segmentation 必须与 ultralytics 的 mask->polygon 转换逐位一致。"""

import cv2
import numpy as np
import pytest

from assistlabel.core.geometry import mask_to_coco_segmentation

ultralytics = pytest.importorskip("ultralytics")
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg  # noqa: E402


def _make_mask(rng: np.random.Generator) -> np.ndarray:
    m = np.zeros((96, 128), np.uint8)
    for _ in range(4):
        cx, cy = rng.integers(20, 108), rng.integers(20, 76)
        r = int(rng.integers(8, 25))
        cv2.circle(m, (int(cx), int(cy)), r, 1, -1)
    cv2.rectangle(m, (4, 4), (30, 90), 1, -1)
    return m


def test_matches_ultralytics_converter(tmp_path):
    rng = np.random.default_rng(7)
    for i in range(3):
        m = _make_mask(rng)
        ours, area = mask_to_coco_segmentation(m.astype(bool))
        assert ours and area > 0

        mask_dir = tmp_path / f"m{i}"
        out_dir = tmp_path / f"o{i}"
        mask_dir.mkdir()
        cv2.imwrite(str(mask_dir / "img.png"), m * 1)
        convert_segment_masks_to_yolo_seg(str(mask_dir), str(out_dir), classes=1)

        line = (out_dir / "img.txt").read_text().splitlines()[0].split()
        ref = [float(v) for v in line[1:]]
        ref_px = [
            round(v * 128 if j % 2 == 0 else v * 96, 4)
            for j, v in enumerate(ref)
        ]
        ours_px = [round(c, 4) for c in ours[0]]
        # COCO 里坐标只保留 2 位小数，按该精度比对
        assert [round(c, 2) for c in ours_px] == [
            round(c, 2) for c in ref_px
        ], f"polygon mismatch on case {i}"
