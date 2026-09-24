"""Performance & reliability tests: batched depth phase, prefetch pipeline,
corrupt-image isolation, --limit, and output verification/repair."""

import cv2
import numpy as np
import pytest
import yaml

from assistlabel.core.config import load_run_config
from assistlabel.core.pipeline import Manifest
from assistlabel.io.dataset import out_paths, scan_images
from assistlabel.io.depth_io import read_depth_png
from assistlabel.cli import app
from assistlabel.runner import run_pipeline
from assistlabel.verify import verify_outputs, repair


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    # mixed orientations to exercise the orientation bucketing
    sizes = [(120, 160, 3), (160, 120, 3), (120, 160, 3), (160, 120, 3)]
    for i, s in enumerate(sizes):
        cv2.imwrite(str(raw / f"img_{i}.png"), rng.integers(0, 255, s, dtype=np.uint8))

    ontology = tmp_path / "ontology.yaml"
    ontology.write_text("classes:\n  car: {prompt: 'car'}\n", encoding="utf-8")
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        yaml.safe_dump(
            {
                "dataset": {"image_dir": "data/raw", "out_dir": "out"},
                "tasks": ["depth"],
                "depth": {
                    "model": "mock-depth",
                    "device": "cpu",
                    "batch_size": 2,       # forces multiple batches over 4 images
                    "prefetch": 2,
                    "write_workers": 2,
                },
                "detect": {"model": "mock-detect", "ontology": "ontology.yaml", "device": "cpu"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return load_run_config(run_yaml)


def test_batched_depth_phase(project):
    report = run_pipeline(project, ["depth"], resume=False)
    assert report.processed == 4 and report.failed == 0
    for image in scan_images(project.dataset.image_dir):
        paths = out_paths(project.dataset.out_dir, image)
        assert paths["depth_png"].exists()
        depth = read_depth_png(paths["depth_png"])
        assert depth.ndim == 2 and depth.max() > 0
        # run 只产标注（16-bit PNG）；可视化由 assistlabel viz 按需生成
        assert not paths["depth_viz"].exists()


def test_corrupt_image_isolated(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(2)
    for i in range(3):
        cv2.imwrite(str(raw / f"ok_{i}.png"), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
    (raw / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\nGARBAGE-NOT-A-IMAGE")

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        yaml.safe_dump(
            {
                "dataset": {"image_dir": "raw", "out_dir": "out"},
                "tasks": ["depth"],
                "depth": {"model": "mock-depth", "device": "cpu", "batch_size": 4},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cfg = load_run_config(run_yaml)
    report = run_pipeline(cfg, ["depth"], resume=False)
    assert report.processed == 3
    assert report.failed == 1
    assert report.failures[0]["image"].endswith("broken.png")

    manifest = Manifest(out_paths(cfg.dataset.out_dir, "x")["manifest"])
    assert manifest.task_status(str(raw / "broken.png"), "depth") == "failed"
    # resume reruns only the failed one
    report2 = run_pipeline(cfg, ["depth"], resume=True)
    assert report2.skipped == 3 and report2.failed == 1  # still broken -> fails again


def test_verify_detects_torn_write_and_repairs(project):
    run_pipeline(project, ["depth"], resume=False)
    images = scan_images(project.dataset.image_dir)

    # simulate a torn write: garbage PNG + missing meta
    broken = out_paths(project.dataset.out_dir, images[1])
    broken["depth_png"].write_bytes(b"torn-write-garbage")

    issues = verify_outputs(project)
    kinds = {(i.task, i.kind) for i in issues}
    assert ("depth", "corrupt") in kinds
    assert all(i.image == str(images[1]) for i in issues)  # others untouched

    n = repair(project, issues)
    # two issues (corrupt png + missing meta) on the same image reset ONE entry
    assert n == len({(i.image, i.task) for i in issues}) == 1
    manifest = Manifest(out_paths(project.dataset.out_dir, "x")["manifest"])
    assert not manifest.is_done(str(images[1]), "depth")

    # reprocess repaired entry; everything passes verification now
    run_pipeline(project, ["depth"], resume=True)
    assert verify_outputs(project) == []


def test_limit_pilot_run(project):
    report = run_pipeline(project, ["depth"], resume=False, limit=2)
    assert report.total == 2 and report.processed == 2
    manifest = Manifest(out_paths(project.dataset.out_dir, "x")["manifest"])
    assert len(manifest.records) == 2


def test_viz_command_generates_previews(project):
    """assistlabel viz：从已有标注产物生成三类可视化。"""
    from typer.testing import CliRunner

    from assistlabel.cli import app

    run_pipeline(project, ["depth", "detect"], resume=False)
    out = project.dataset.out_dir
    assert not (out / "depth_viz").exists()  # run 不产 viz

    run_yaml = project.dataset.out_dir.parent / "run.yaml"
    r = CliRunner().invoke(app, ["viz", "--config", str(run_yaml)])
    assert r.exit_code == 0

    for d in ("depth_viz", "detect_viz", "semantic_viz"):
        files = list((out / d).glob("*"))
        assert files, f"{d} 未生成"


def test_viz_kind_independent(tmp_path):
    """每种可视化可独立生成，互不影响。"""
    import cv2 as _cv2
    from typer.testing import CliRunner

    from assistlabel.cli import app

    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(7)
    cv2.imwrite(str(raw / "solo.png"), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
    onto = tmp_path / "ontology.yaml"
    onto.write_text("classes:\n  car: {prompt: 'car'}\n", encoding="utf-8")
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump({
        "dataset": {"image_dir": "raw", "out_dir": "out"},
        "tasks": ["depth", "detect"],
        "depth": {"model": "mock-depth", "device": "cpu"},
        "detect": {"model": "mock-detect", "ontology": "ontology.yaml", "device": "cpu"},
    }), encoding="utf-8")

    runner = CliRunner()
    run_pipeline(load_run_config(run_yaml), ["depth", "detect"], resume=False)

    # 只生成 depth 可视化
    r = runner.invoke(app, ["viz", "-c", str(run_yaml), "--kind", "depth"])
    assert r.exit_code == 0
    out = tmp_path / "out"
    assert (out / "depth_viz").exists()
    assert not (out / "detect_viz").exists()  # 未指定则不生成

    # 补生成 detect 可视化
    r = runner.invoke(app, ["viz", "-c", str(run_yaml), "--kind", "detect"])
    assert r.exit_code == 0
    assert (out / "detect_viz").exists()
