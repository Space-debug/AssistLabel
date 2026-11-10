"""Manifest/annotations.json consistency.

Labels are journaled per image before the manifest marks it ``done``; the
aggregate annotations.json is rebuilt at the end of the detect phase. A crash
must therefore leave every ``done`` record backed by a journal fragment, and a
resume must not duplicate annotations.
"""

import json
import os

import cv2
import numpy as np
import pytest
import yaml

from assistlabel import runner
from assistlabel.core.config import load_run_config
from assistlabel.core.pipeline import Manifest
from assistlabel.io.annotations import fragment_store_path
from assistlabel.io.dataset import out_paths


def _project(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        cv2.imwrite(str(raw / f"img{i}.png"), np.full((40, 60, 3), i * 20, np.uint8))
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        yaml.safe_dump({
            "dataset": {"image_dir": "raw", "out_dir": "out"},
            "tasks": ["detect"],
            "detect": {"model": "mock-detect", "device": "cpu",
                       "classes": {"car": "car"}},
        }, sort_keys=False),
        encoding="utf-8",
    )
    return load_run_config(run_yaml)


def _journal(cfg):
    path = fragment_store_path(cfg.dataset.out_dir / "annotations.json")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _relative_names(cfg, keys):
    return {
        os.path.relpath(k, cfg.dataset.image_dir).replace("\\", "/") for k in keys
    }


def test_done_implies_fragment_persisted(tmp_path, monkeypatch):
    cfg = _project(tmp_path)

    def boom(_coco, _path):
        raise RuntimeError("disk full")

    # the aggregate file fails to write; journaled fragments must still exist
    monkeypatch.setattr(runner, "coco_save", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        runner.run_pipeline(cfg, ["detect"], resume=False)

    manifest = Manifest(out_paths(cfg.dataset.out_dir, "x")["manifest"])
    done = [
        key for key, rec in manifest.records.items()
        if rec.get("tasks", {}).get("detect", {}).get("status") == "done"
    ]
    assert done, "mock pipeline should have completed some images"
    assert _relative_names(cfg, done) <= {f["image"]["file_name"] for f in _journal(cfg)}


def test_resume_after_crash_has_no_duplicates(tmp_path, monkeypatch):
    cfg = _project(tmp_path)

    def boom(_coco, _path):
        raise RuntimeError("disk full")

    monkeypatch.setattr(runner, "coco_save", boom)
    with pytest.raises(RuntimeError):
        runner.run_pipeline(cfg, ["detect"], resume=False)

    monkeypatch.undo()  # "restart" with a healthy disk
    report = runner.run_pipeline(cfg, ["detect"], resume=True)
    assert report.failed == 0

    coco = json.loads((cfg.dataset.out_dir / "annotations.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 3  # one mock object per image, no duplicates
    assert len({a["id"] for a in coco["annotations"]}) == 3


def test_successful_run_compacts_journal(tmp_path):
    cfg = _project(tmp_path)
    report = runner.run_pipeline(cfg, ["detect"], resume=False)
    assert report.failed == 0

    manifest = Manifest(out_paths(cfg.dataset.out_dir, "x")["manifest"])
    assert manifest.summary()["tasks"]["detect"]["done"] == 3

    coco = json.loads((cfg.dataset.out_dir / "annotations.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 3
    assert len(_journal(cfg)) == 3  # compacted to one line per image
