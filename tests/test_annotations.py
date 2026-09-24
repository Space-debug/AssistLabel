import numpy as np
import pytest

from assistlabel.core.engine import Detection
from assistlabel.io.annotations import (
    build_labelme,
    coco_empty_store,
    coco_upsert_image,
    detection_to_shapes,
    export_ultralytics,
    load_labelme,
    rle_decode,
    rle_encode,
    save_labelme,
)


def _det(mask=True):
    m = None
    if mask:
        m = np.zeros((50, 60), dtype=bool)
        m[10:40, 20:50] = True
    return Detection(label="car", score=0.87, box=(20, 10, 50, 40), mask=m)


def test_detection_to_shapes_polygon():
    shapes = detection_to_shapes(_det(), mask_to="polygon")
    assert shapes and shapes[0]["shape_type"] == "polygon"
    assert shapes[0]["label"] == "car"
    assert shapes[0]["score"] == 0.87
    assert len(shapes[0]["points"]) >= 3


def test_detection_to_shapes_box_fallback():
    shapes = detection_to_shapes(_det(mask=False), mask_to="polygon")
    assert shapes[0]["shape_type"] == "rectangle"
    assert len(shapes[0]["points"]) == 4


def test_detection_to_shapes_rle():
    shapes = detection_to_shapes(_det(), mask_to="rle")
    assert len(shapes) == 1
    assert shapes[0]["shape_type"] == "mask"
    assert shapes[0]["rle"]["size"] == [50, 60]
    m = rle_decode(shapes[0]["rle"])
    assert m[20, 30] and not m[0, 0]  # inside mask / outside mask


def test_labelme_roundtrip(tmp_path):
    shapes = detection_to_shapes(_det(), mask_to="polygon")
    data = build_labelme("img.jpg", (60, 50), shapes)
    p = tmp_path / "img.json"
    save_labelme(data, p)
    loaded = load_labelme(p)
    assert loaded["imageWidth"] == 60 and loaded["imageHeight"] == 50
    assert loaded["shapes"][0]["label"] == "car"


def test_rle_roundtrip_preserves_mask():
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:15, 10:30] = True
    mask[20:25, 5:8] = True
    rle = rle_encode(mask)
    assert rle_decode(rle).tolist() == mask.tolist()


def _write_coco_store(tmp_path):
    """Minimal COCO store: one image with one RLE annotation."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:70] = True
    store = coco_empty_store(["car"])
    iid = coco_upsert_image(store, "a.jpg", 100, 100)
    store["annotations"].append({
        "image_id": iid,
        "category_id": 1,
        "segmentation": rle_encode(mask),
        "bbox": [30.0, 20.0, 40.0, 40.0],
        "area": int(mask.sum()),
        "iscrowd": 0,
        "score": 0.9,
    })
    json_path = tmp_path / "detect" / "annotations.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    json_path.write_text(json.dumps(store), encoding="utf-8")
    return json_path


def test_export_ultralytics_from_coco(tmp_path):
    import cv2

    img_path = tmp_path / "a.jpg"
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), np.uint8))
    json_path = _write_coco_store(tmp_path)

    out = tmp_path / "ultralytics"
    stats = export_ultralytics(json_path, out, image_dir=tmp_path, split=0.8)
    assert stats["train"] + stats["val"] == 1

    bucket = "train" if stats["train"] else "val"
    assert (out / "images" / bucket / "a.jpg").exists()
    lines = (out / "labels" / bucket / "a.txt").read_text().strip().splitlines()
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"  # category_id 1 -> YOLO class 0
    assert all(0.0 < float(v) < 1.0 for v in (cx, cy, w, h))
    assert f"nc: 1" in (out / "data.yaml").read_text()
