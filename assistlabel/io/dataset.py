"""Image discovery, content hashing and output-path conventions.

Output layout (per PLAN.md):

    out_dir/
      manifest.jsonl
      images/            copies of source images
      depth/             16-bit PNG + *_meta.json + *_color.jpg
      labels/            labelme .json per image
      viz/               detection overlays for eyeballing
      coco/              export target
"""

from __future__ import annotations

import hashlib
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def scan_images(image_dir: str | Path, patterns: list[str] | None = None) -> list[Path]:
    """Recursively discover images under ``image_dir``.

    ``patterns`` are glob patterns (e.g. ``*.jpg``); when given, they override
    the default suffix list. Results are sorted for deterministic runs.
    """
    root = Path(image_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"image_dir does not exist: {root}")

    if patterns:
        seen: set[Path] = set()
        out: list[Path] = []
        for pat in patterns:
            for p in root.rglob(pat):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p not in seen:
                    seen.add(p)
                    out.append(p)
        return sorted(out, key=lambda p: str(p).lower())

    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p).lower(),
    )


def file_hash(path: str | Path, head_bytes: int = 64 * 1024) -> str:
    """Fast, stable content hash: sha1 of (file size + first head_bytes)."""
    path = Path(path)
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(size.to_bytes(8, "little"))
    with open(path, "rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()[:16]


def out_paths(out_dir: str | Path, image_path: str | Path, image_dir: str | Path | None = None):
    """Canonical output paths for one source image.

    Returns a dict of Paths keyed by role. When ``image_dir`` is given and the image lives
    in a subdirectory of it, the relative path is flattened into the stem with
    ``__`` separators (``sub/img.png`` -> ``sub__img``) so recursively-scanned
    datasets cannot collide on output files. Without ``image_dir`` (or for
    top-level images) the plain stem is used.
    """
    out_dir = Path(out_dir)
    image_path = Path(image_path)
    stem = image_path.stem
    if image_dir is not None:
        try:
            rel = image_path.resolve().relative_to(Path(image_dir).resolve())
            if len(rel.parts) > 1:
                stem = "__".join(rel.with_suffix("").parts)
        except (ValueError, OSError):
            pass  # image not under image_dir: fall back to plain stem
    return {
        "manifest": out_dir / "manifest.jsonl",
        "detect_json": out_dir / "detect" / "annotations.json",
        "semantic_png": out_dir / "semantic" / f"{stem}.png",
        "semantic_classes": out_dir / "semantic" / "classes.txt",
        "semantic_viz": out_dir / "semantic_viz" / f"{stem}.jpg",
        "depth_png": out_dir / "depth" / f"{stem}.png",
        "depth_viz": out_dir / "depth_viz" / f"{stem}.jpg",
        "detect_viz": out_dir / "detect_viz" / f"{stem}.jpg",
        "ultralytics_dir": out_dir / "ultralytics",
    }
