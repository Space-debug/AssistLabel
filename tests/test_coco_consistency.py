"""Manifest/annotations.json consistency.

A ``done`` detect record must never exist for an image whose labels were not
persisted: otherwise a crash makes resume skip that image and the labels are
lost silently.
"""

import cv2
import numpy as np
import pytest
import yaml

from assistlabel import runner
from assistlabel.core.config import load_run_config
from assistlabel.core.pipeline import Manifest
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


def test_annotations_failure_never_publishes_done(tmp_path, monkeypatch):
    cfg = _project(tmp_path)

    def boom(_coco, _path):
        raise RuntimeError("disk full")

    monkeypatch.setattr(runner, "coco_save", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        runner.run_pipeline(cfg, ["detect"], resume=False)

    manifest = Manifest(out_paths(cfg.dataset.out_dir, "x")["manifest"])
    done = manifest.summary()["tasks"].get("detect", {}).get("done", 0)
    assert done == 0, "done records must not exist before labels are persisted"


def test_successful_run_marks_done_after_labels_exist(tmp_path):
    cfg = _project(tmp_path)
    report = runner.run_pipeline(cfg, ["detect"], resume=False)
    assert report.failed == 0

    manifest = Manifest(out_paths(cfg.dataset.out_dir, "x")["manifest"])
    done = manifest.summary()["tasks"]["detect"]["done"]
    assert done == 3

    import json

    coco = json.loads((cfg.dataset.out_dir / "annotations.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == 3
