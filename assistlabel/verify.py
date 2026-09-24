"""Output integrity verification: manifest claims vs files actually on disk.

A crash / power cut can leave a task marked done while its output PNG or JSON
is torn. ``verify_outputs`` re-reads every artifact a "done" record claims;
``repair`` clears the broken manifest entries so the next ``run --resume``
reprocesses exactly those images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pathlib import Path

import cv2
import numpy as np

from .core.config import RunConfig
from .core.pipeline import STATUS_DONE, Manifest
from .io.dataset import file_hash, out_paths
from .io.depth_io import imread_unchanged


@dataclass
class Issue:
    image: str
    task: str
    kind: str  # missing | corrupt | empty
    detail: str


def _manifest(cfg: RunConfig) -> Manifest:
    return Manifest(out_paths(cfg.dataset.out_dir, "x")["manifest"])


def verify_outputs(cfg: RunConfig) -> list[Issue]:
    manifest = _manifest(cfg)
    issues: list[Issue] = []
    for key, rec in manifest.records.items():
        tasks = rec.get("tasks", {})
        if tasks.get("depth", {}).get("status") == STATUS_DONE:
            issues.extend(_verify_depth(cfg, key, rec))
        if tasks.get("detect", {}).get("status") == STATUS_DONE:
            issues.extend(_verify_detect(cfg, key))
    return issues


def _verify_depth(cfg: RunConfig, key: str, rec: dict) -> list[Issue]:
    issues: list[Issue] = []
    paths = out_paths(cfg.dataset.out_dir, key, cfg.dataset.image_dir)
    png = paths["depth_png"]
    if not png.exists():
        return [Issue(key, "depth", "missing", f"{png.name} not found")]
    raw = imread_unchanged(png)
    if raw is None:
        issues.append(Issue(key, "depth", "corrupt", f"{png.name} is not readable"))
    elif raw.dtype != np.uint16:
        issues.append(Issue(key, "depth", "corrupt", f"expected uint16, got {raw.dtype}"))
    elif not (raw > 0).any():
        issues.append(Issue(key, "depth", "empty", "all pixels invalid (0)"))
    # P2-3: source image content changed after labeling -> annotations stale
    stored_hash = rec.get("tasks", {}).get("depth", {}).get("source_image_hash")
    source = Path(key)
    if stored_hash and source.exists() and file_hash(source) != stored_hash:
        issues.append(Issue(key, "depth", "stale", "source image changed since labeling"))
    return issues


def _verify_detect(cfg: RunConfig, key: str) -> list[Issue]:
    paths = out_paths(cfg.dataset.out_dir, key, cfg.dataset.image_dir)
    sem = paths["semantic_png"]
    if not sem.exists():
        return [Issue(key, "detect", "missing", f"{sem.name} not found")]
    raw = imread_unchanged(sem)
    if raw is None:
        return [Issue(key, "detect", "corrupt", f"{sem.name} is not readable")]
    if raw.dtype != np.uint8:
        return [Issue(key, "detect", "corrupt", f"expected uint8, got {raw.dtype}")]
    return []


def repair(cfg: RunConfig, issues: list[Issue]) -> int:
    """Clear broken manifest entries; returns how many were reset.

    Cleared tasks are re-executed by the next ``assistlabel run --resume``.
    """
    if not issues:
        return 0
    manifest = _manifest(cfg)
    changed = 0
    for issue in issues:
        rec = manifest.records.get(issue.image)
        if rec and issue.task in rec.get("tasks", {}):
            del rec["tasks"][issue.task]
            changed += 1
    manifest.save()
    return changed
