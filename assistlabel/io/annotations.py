"""Annotation I/O: COCO (detect 原生产物) + 语义 PNG + RLE 工具。

detect 任务直接产出：
- ``detect/annotations.json``  COCO（RLE 实例掩码 + bbox + score + 深度统计）
- ``semantic/<stem>.png``      class-index 语义掩码（0=背景）

本模块提供：RLE 编解码、COCO store 增删存取、ultralytics 训练布局导出、
以及 labelme 兼容读写（互操作用）。
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from ..core.engine import Detection
from ..core.geometry import mask_to_polygons

LABELME_VERSION = "5.4.1"


# ---------------------------------------------------------------------------
# COCO store utilities (detect 任务的原生输出格式)
# ---------------------------------------------------------------------------
def coco_load(path: str | Path) -> dict:
    """Load an existing COCO dict (or an empty one) with stable sections."""
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            coco = json.load(f)
    else:
        coco = {}
    coco.setdefault("images", [])
    coco.setdefault("annotations", [])
    coco.setdefault("categories", [])
    return coco


def coco_save(coco: dict, path: str | Path) -> None:
    """Atomically write a COCO dict (tmp + rename, like the manifest)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".coco_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def coco_empty_store(class_names: list[str]) -> dict:
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": ""}
            for i, name in enumerate(class_names)
        ],
    }


def coco_upsert_image(coco: dict, file_name: str, width: int, height: int) -> int:
    """Insert or replace an image entry; drops that image's old annotations.

    Returns the (possibly new) image_id.
    """
    for im in coco["images"]:
        if im["file_name"] == file_name:
            im["width"], im["height"] = int(width), int(height)
            coco["annotations"] = [
                a for a in coco["annotations"] if a["image_id"] != im["id"]
            ]
            return im["id"]
    nid = max((im["id"] for im in coco["images"]), default=0) + 1
    coco["images"].append(
        {"id": nid, "file_name": file_name, "width": int(width), "height": int(height)}
    )
    return nid


def coco_next_annotation_id(coco: dict) -> int:
    return max((a["id"] for a in coco["annotations"]), default=0) + 1


# ---------------------------------------------------------------------------
# RLE helpers (COCO uncompressed RLE, 纯标准库编解码)
# ---------------------------------------------------------------------------
def rle_encode(mask: np.ndarray) -> dict:
    """bool (H, W) mask -> COCO uncompressed RLE: {"size": [h, w], "counts": [int]}.

    Column-major runs starting with a background run (pycocotools convention);
    plain JSON-serializable ints, no extra dependency.
    """
    mask = np.asarray(mask) > 0
    h, w = mask.shape
    flat = mask.flatten(order="F").astype(np.uint8)
    changes = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate([[0], changes, [flat.size]])
    counts = np.diff(boundaries).tolist()
    if flat.size and flat[0] == 1:
        counts = [0] + counts
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def rle_decode(rle: dict) -> np.ndarray:
    """COCO uncompressed RLE -> bool (H, W) mask."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in rle["counts"]:
        flat[pos : pos + int(c)] = val
        pos += int(c)
        val = 1 - val
    return flat.reshape((w, h)).T.astype(bool)


# ---------------------------------------------------------------------------
# labelme interop（可编辑工作格式的读写；当前 run 直接产 COCO，不再经过它）
# ---------------------------------------------------------------------------
def detection_to_shapes(det: Detection, mask_to: str = "polygon") -> list[dict]:
    """One Detection -> list of labelme shape dicts.

    - polygon (default): simplified contours — compact and X-AnyLabeling
      editable, but edges are ~95% IoU vs the raw mask
    - rle: pixel-exact mask stored as shape_type "mask" with COCO uncompressed
      RLE; not X-AnyLabeling-editable
    """
    shapes: list[dict] = []
    if det.mask is not None and mask_to == "polygon":
        for poly in mask_to_polygons(det.mask):
            shapes.append(
                {
                    "label": det.label,
                    "points": [[float(x), float(y)] for x, y in poly],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                    "score": float(det.score),
                    "extra": dict(det.extra),
                }
            )
    elif det.mask is not None and mask_to == "rle":
        shapes.append(
            {
                "label": det.label,
                "points": [],
                "group_id": None,
                "shape_type": "mask",
                "rle": rle_encode(det.mask),
                "flags": {},
                "score": float(det.score),
                "extra": dict(det.extra),
            }
        )
    if not shapes:  # box-only (no mask or conversion unusable)
        x1, y1, x2, y2 = [float(v) for v in det.box]
        shapes.append(
            {
                "label": det.label,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "score": float(det.score),
                "extra": dict(det.extra),
            }
        )
    return shapes


def build_labelme(
    image_filename: str,
    image_size: tuple[int, int],
    shapes: list[dict],
    image_path: str | None = None,
) -> dict:
    w, h = image_size
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path or image_filename,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def save_labelme(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_labelme(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ultralytics 训练布局导出（从 detect 的 COCO store 派生）
# ---------------------------------------------------------------------------
def export_ultralytics(
    coco_json: str | Path,
    out_dir: str | Path,
    image_dir: str | Path,
    split: float = 0.8,
    seed: int = 0,
    copy_fallback: bool = True,
) -> dict:
    """Write a ready-to-train ultralytics dataset layout from the detect
    COCO store::

        out_dir/
          data.yaml
          images/{train,val}/...   (hard links; falls back to copies)
          labels/{train,val}/...   (YOLO txt: class cx cy w h, normalized)

    Train/val assignment is a seeded random split over images. Returns a
    stats dict: {"train": n, "val": n}.
    """
    import shutil

    coco = coco_load(coco_json)
    out_dir = Path(out_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    ordered_ids = sorted(cats)
    names = [cats[cid] for cid in ordered_ids]
    (out_dir / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names!r}\n",
        encoding="utf-8",
    )

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    rng = random.Random(seed)
    stats = {"train": 0, "val": 0}
    image_dir = Path(image_dir)
    for im in coco.get("images", []):
        bucket = "train" if rng.random() < split else "val"
        w, h = float(im.get("width", 0)), float(im.get("height", 0))
        lines = []
        ordered_ids = sorted(cats)
        yolo_id = {cid: i for i, cid in enumerate(ordered_ids)}  # 0-based
        for ann in anns_by_image.get(im["id"], []):
            x, y, bw, bh = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
                continue
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            lines.append(
                f"{yolo_id[ann['category_id']]} "
                f"{cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}"
            )
        stem = Path(im["file_name"]).stem
        dst_label = out_dir / "labels" / bucket / f"{stem}.txt"
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        src_img = image_dir / im["file_name"]
        dst_img = out_dir / "images" / bucket / Path(im["file_name"]).name
        if dst_img.exists():
            dst_img.unlink()  # allow re-export over a previous run
        if src_img.exists():
            try:
                os.link(src_img, dst_img)  # same volume: zero-copy
            except OSError:  # cross-volume or locked -> copy
                shutil.copy2(src_img, dst_img)
        stats[bucket] += 1
    return stats
