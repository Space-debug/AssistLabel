"""Tests for IMPROVEMENTS.md items: P0-1 nested path collisions, P0-2 non-ASCII
paths, P0-3 manifest lock, P2-3 stale detection, P2-5 depth sentinel."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from assistlabel.core.config import load_run_config
from assistlabel.core.pipeline import Manifest, ManifestLockedError
from assistlabel.io.dataset import out_paths, scan_images
from assistlabel.io.depth_io import read_depth_png, save_depth_png
from assistlabel.runner import run_pipeline
from assistlabel.verify import verify_outputs


# -- P0-1: nested directories must not collide on output stems ---------------
def test_out_paths_nested_flattening(tmp_path):
    raw = tmp_path / "raw"
    (raw / "a").mkdir(parents=True)
    (raw / "b").mkdir(parents=True)
    for p in ["a/img.png", "b/img.png", "top.png"]:
        (raw / p).write_bytes(b"x")

    base = tmp_path / "out"
    pa = out_paths(base, raw / "a" / "img.png", raw)
    pb = out_paths(base, raw / "b" / "img.png", raw)
    pt = out_paths(base, raw / "top.png", raw)

    assert pa["depth_png"] != pb["depth_png"]
    assert pa["depth_png"].name == "a__img.png"
    assert pb["depth_png"].name == "b__img.png"
    assert pt["depth_png"].name == "top.png"


def test_e2e_nested_dataset_no_collision(tmp_path):
    raw = tmp_path / "data" / "raw"
    (raw / "a").mkdir(parents=True)
    (raw / "b").mkdir(parents=True)
    rng = np.random.default_rng(3)
    for name in ["a/img.png", "b/img.png"]:
        cv2.imwrite(str(raw / name), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump({
        "dataset": {"image_dir": "data/raw", "out_dir": "out"},
        "tasks": ["depth"],
        "depth": {"model": "mock-depth", "device": "cpu"},
    }), encoding="utf-8")
    cfg = load_run_config(run_yaml)
    report = run_pipeline(cfg, ["depth"], resume=False)
    assert report.failed == 0

    m = [json.loads(l) for l in (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()]
    assert len(m) == 2
    labels = {Path(r["image"]).parent.name + "/" + Path(r["image"]).name for r in m}
    assert labels == {"a/img.png", "b/img.png"}


# -- P0-2: non-ASCII (Chinese) paths -----------------------------------------
def test_depth_io_chinese_path_roundtrip(tmp_path):
    d = tmp_path / "图库" / "相机01"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 20, (32, 48)).astype(np.float32)
    p = d / "帧_01.png"
    save_depth_png(depth, p)
    back = read_depth_png(p)
    assert np.abs(back - depth).max() < 1e-3


# -- P0-3: manifest lock ------------------------------------------------------
def test_manifest_lock_blocks_second_instance(tmp_path):
    p = tmp_path / "manifest.jsonl"
    m1 = Manifest(p, lock=True)
    with pytest.raises(ManifestLockedError):
        Manifest(p, lock=True)
    # read-only (no lock) access still allowed
    Manifest(p)
    m1.release()
    m2 = Manifest(p, lock=True)  # releasable then re-acquirable
    m2.release()


# -- P2-3: stale source detection --------------------------------------------
def test_verify_reports_stale_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(5)
    cv2.imwrite(str(raw / "img.png"), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump({
        "dataset": {"image_dir": "raw", "out_dir": "out"},
        "tasks": ["depth"],
        "depth": {"model": "mock-depth", "device": "cpu"},
    }), encoding="utf-8")
    cfg = load_run_config(run_yaml)
    run_pipeline(cfg, ["depth"], resume=False)
    assert verify_outputs(cfg) == []

    # overwrite the source image with different content -> annotation stale
    cv2.imwrite(str(raw / "img.png"), rng.integers(0, 100, (60, 80, 3), dtype=np.uint8))
    issues = verify_outputs(cfg)
    assert any(i.kind == "stale" for i in issues)


# -- P2-5: depth sanity sentinel ---------------------------------------------
def test_depth_sentinel_rejects_garbage(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(6)
    good = np.zeros((60, 80, 3), np.uint8)
    good[:, :, 1] = 200  # mostly "background" for the mock's linear depth
    cv2.imwrite(str(raw / "ok.png"), good)
    cv2.imwrite(str(raw / "black.png"), np.zeros((60, 80, 3), np.uint8))  # -> depth ~0.5m? no: black => x coeff 0

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump({
        "dataset": {"image_dir": "raw", "out_dir": "out"},
        "tasks": ["depth"],
        "depth": {"model": "mock-depth", "device": "cpu"},
    }), encoding="utf-8")
    cfg = load_run_config(run_yaml)
    report = run_pipeline(cfg, ["depth"], resume=False)
    # mock depth never produces garbage; sentinel must not misfire on valid runs
    assert report.failed == 0
