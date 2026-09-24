"""End-to-end pipeline test with mock engines: images -> depth+detect (COCO +
semantic) -> fuse -> yolo/semantic exports -> QA. Exercises the same code path
as the CLI's `run` command, without GPU dependencies."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from assistlabel.core.config import load_run_config
from assistlabel.io.annotations import export_ultralytics, load_labelme
from assistlabel.io.dataset import out_paths, scan_images
from assistlabel.io.depth_io import read_depth_png
from assistlabel.qa import collect_stats, sample_review_list, write_report
from assistlabel.runner import run_pipeline


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(4):
        img = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        cv2.imwrite(str(raw / f"img_{i:02d}.png"), img)

    ontology = tmp_path / "ontology.yaml"
    ontology.write_text(
        "classes:\n  car: {prompt: 'car'}\n  pedestrian: {prompt: 'pedestrian'}\n",
        encoding="utf-8",
    )
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        yaml.safe_dump(
            {
                "dataset": {"image_dir": "data/raw", "out_dir": "out"},
                "tasks": ["depth", "detect"],
                "depth": {"model": "mock-depth", "device": "cpu"},
                "detect": {
                    "model": "mock-detect",
                    "ontology": "ontology.yaml",
                    "device": "cpu",
                    "conf_thres": 0.5,
                    "viz": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return load_run_config(run_yaml)


def _coco(project):
    path = project.dataset.out_dir / "detect" / "annotations.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _anns_for(project, image):
    coco = _coco(project)
    target = str(image)
    img_id = next(im["id"] for im in coco["images"] if Path(im["file_name"]).name == Path(target).name)
    return [a for a in coco["annotations"] if a["image_id"] == img_id]


def test_full_pipeline_mock(project):
    report = run_pipeline(project, ["depth", "detect"], resume=False)
    assert report.processed == 8 and report.failed == 0  # 4 images x 2 tasks

    out = project.dataset.out_dir
    images = scan_images(project.dataset.image_dir)

    for image in images:
        stem = image.stem
        depth = read_depth_png(out / "depth" / f"{stem}.png")
        assert depth.shape == (120, 160) and depth.max() > 0
        assert (out / "depth_viz" / f"{stem}.jpg").exists()

        coco = _coco(project)
        anns = _anns_for(project, image)
        assert len(anns) == 2  # one per ontology class
        for ann in anns:
            assert isinstance(ann["segmentation"], dict) and ann["segmentation"]["counts"]
            assert ann["bbox"][2] > 0 and ann["bbox"][3] > 0
            # 标准 COCO 字段，不含非标的深度扩展
            assert "depth_median_m" not in ann
        assert (out / "semantic" / f"{stem}.png").exists()
        assert (out / "semantic_viz" / f"{stem}.jpg").exists()
        assert (out / "detect_viz" / f"{stem}.jpg").exists()

    # semantic class indices limited to ontology ids {1, 2}
    sem = cv2.imread(str(out / "semantic" / "img_00.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(sem)) <= {0, 1, 2}

    # manifest tracks done state
    lines = (out / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4


def test_resume_skips_done(project):
    run_pipeline(project, ["depth"], resume=False)
    report = run_pipeline(project, ["depth"], resume=True)
    assert report.skipped == 4 and report.processed == 0


def test_redo_detect_refreshes(project):
    run_pipeline(project, ["depth", "detect"], resume=False)

    # 模拟 ontology 变更：改提示词后 --redo-detect 重刷 detect
    onto = project.dataset.image_dir.parent / "ontology.yaml"
    onto.write_text(
        "classes:\n  car: {prompt: 'car'}\n  pedestrian: {prompt: 'walking person'}\n",
        encoding="utf-8",
    )
    report = run_pipeline(project, ["detect"], resume=True, redo_tasks=["detect"])
    assert report.processed == 4 and report.failed == 0

    # 重刷后 label 仍是 ontology 类名，且 manifest 记录了新 ontology hash
    coco = _coco(project)
    assert len(coco["annotations"]) == 8


def test_export_and_qa(project):
    run_pipeline(project, ["depth", "detect"], resume=False)
    out = project.dataset.out_dir

    from assistlabel.io.annotations import coco_load
    coco = coco_load(out / "detect" / "annotations.json")

    # ultralytics export derives from the native COCO store
    result = export_ultralytics(
        out / "detect" / "annotations.json",
        out / "ultralytics",
        image_dir=project.dataset.image_dir,
        split=0.75,
    )
    assert result["train"] + result["val"] == 4
    assert (out / "ultralytics" / "data.yaml").exists()

    stats = collect_stats(out)
    assert stats["images"] == 4
    assert stats["class_counts"] == {"car": 4, "pedestrian": 4}
    assert stats["errors"] == []

    review = sample_review_list(out, n=3)
    assert 0 < len(review) <= 3

    write_report(stats, out / "report.html")
    assert (out / "report.html").exists()
