"""Depth batching keys: mixed processed sizes must never share a batch.

DA3's InputProcessor center-crops every member of a batch to the smallest
processed (H, W); batching different aspect ratios would corrupt depth
geometry. ``batch_key`` predicts the processed shape and must match it.
"""

import numpy as np

from assistlabel.engines.depth_anything3 import DepthAnything3Engine
from assistlabel.engines.mock import MockDepthEngine


def _img(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_batch_key_groups_different_sizes_with_same_processed_shape():
    engine = DepthAnything3Engine("depth-anything/DA3METRIC-LARGE")
    # 16:9 at two resolutions -> both process to 504x280
    assert engine.batch_key(_img(1080, 1920)) == engine.batch_key(_img(720, 1280))


def test_batch_key_separates_different_aspect_ratios():
    engine = DepthAnything3Engine("depth-anything/DA3METRIC-LARGE")
    wide = engine.batch_key(_img(1080, 1920))   # 16:9
    four_three = engine.batch_key(_img(1200, 1600))  # 4:3
    assert wide != four_three


def test_batch_key_separates_portrait_and_landscape():
    engine = DepthAnything3Engine("depth-anything/DA3METRIC-LARGE")
    assert engine.batch_key(_img(1920, 1080)) != engine.batch_key(_img(1080, 1920))


def test_base_batch_key_falls_back_to_input_shape():
    # engines without internal resizing (mock, DA2) batch on exact input size
    mock = MockDepthEngine()
    assert mock.batch_key(_img(120, 160)) == (120, 160)
    assert mock.batch_key(_img(120, 160)) != mock.batch_key(_img(160, 120))
