"""Depth map I/O: 16-bit PNG (mm) primary format, sidecar metadata,
pseudo-color visualization. KITTI-style convention: depth_m = pixel / 1000,
invalid = 0.

All file I/O goes through encode/tofile + fromfile/imdecode: plain
cv2.imread/imwrite fail silently on Windows paths containing non-ASCII
characters (e.g. Chinese dataset folder names).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

DEPTH_SCALE = 1000.0  # mm per meter
MAX_UINT16 = 65535  # clip at 65.5 m


def quantize_m_to_uint16(depth_m: np.ndarray) -> np.ndarray:
    """Float meters -> uint16 millimeters, invalid (NaN/negative) -> 0."""
    d = np.asarray(depth_m, dtype=np.float32)
    mm = np.where(np.isfinite(d) & (d > 0), d * DEPTH_SCALE, 0.0)
    return np.clip(np.rint(mm), 0, MAX_UINT16).astype(np.uint16)


def dequantize_uint16_to_m(depth_u16: np.ndarray) -> np.ndarray:
    """uint16 millimeters -> float32 meters; zeros stay zero (invalid)."""
    return np.asarray(depth_u16, dtype=np.float32) / DEPTH_SCALE


def imread_unchanged(path: str | Path) -> np.ndarray | None:
    """cv2.imread equivalent that works on Windows non-ASCII paths.

    Returns None when the file cannot be decoded (mirrors cv2 behavior).
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def imwrite_safe(path: str | Path, img: np.ndarray, params: list | None = None) -> None:
    """cv2.imwrite equivalent that works on Windows non-ASCII paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img, params or [])
    if not ok:
        raise IOError(f"Failed to encode image: {path}")
    buf.tofile(str(path))


def save_depth_png(depth_m: np.ndarray, path: str | Path, compression: int = 3) -> None:
    """Write depth as 16-bit PNG (uint16 millimeters).

    ``compression`` maps to cv2.IMWRITE_PNG_COMPRESSION (0=fastest/largest,
    9=smallest/slowest); the default 3 trades ~10% size for ~2x write speed.
    """
    imwrite_safe(path, quantize_m_to_uint16(depth_m),
                 [int(cv2.IMWRITE_PNG_COMPRESSION), int(compression)])


def read_depth_png(path: str | Path) -> np.ndarray:
    """Read a 16-bit depth PNG back as float32 meters (invalid = 0)."""
    raw = imread_unchanged(path)
    if raw is None:
        raise IOError(f"Failed to read depth PNG: {path}")
    if raw.ndim == 3:  # defensive: some writers store 3x8bit
        raw = raw[:, :, 0]
    if raw.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 depth PNG, got {raw.dtype}")
    return dequantize_uint16_to_m(raw)


def save_depth_meta(path: str | Path, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"depth_unit": "mm", "scale": DEPTH_SCALE, **meta}, f, ensure_ascii=False, indent=2)


def pseudocolor(depth_m: np.ndarray, valid_only: bool = True) -> np.ndarray:
    """Turbo pseudo-color visualization (BGR uint8). Normalization uses the
    1st-99th percentile of valid pixels so outliers don't wash out the image."""
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid_only and valid.any():
        lo, hi = np.percentile(d[valid], [1, 99])
        if hi - lo < 1e-6:
            hi = lo + 1.0
    else:
        lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    norm[~valid] = 0.0
    u8 = (norm * 255).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(u8, colormap)


def save_depth_color(depth_m: np.ndarray, path: str | Path) -> None:
    imwrite_safe(path, pseudocolor(depth_m))
