import pytest

from assistlabel.core.config import RunConfig, load_run_config
from assistlabel.core.pipeline import Manifest, run_image_tasks
from assistlabel.core.registry import KIND_DEPTH, ModelRegistry


# -- registry ---------------------------------------------------------------
def test_registry_builtin_models_load():
    registry = ModelRegistry.load()
    assert "sam3" in registry.keys()
    assert "da2-metric-indoor-L" in registry.keys()
    depth_models = registry.filter_kind(KIND_DEPTH)
    assert any(m.key.startswith("da2") for m in depth_models)


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError):
        ModelRegistry.load().get("nope")


def test_registry_extra_dir_overrides(tmp_path):
    (tmp_path / "custom.yaml").write_text(
        "models:\n  - key: sam3\n    provider: mock_detect\n    kind: detect_segment\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.load(extra_dirs=[tmp_path])
    assert registry.get("sam3").provider == "mock_detect"


# -- manifest / resume ------------------------------------------------------
def test_manifest_roundtrip_and_torn_line(tmp_path):
    path = tmp_path / "manifest.jsonl"
    m = Manifest(path)
    m.mark("a.jpg", "depth", "done", seconds=1.0)
    m.mark("a.jpg", "detect", "failed", error="boom")
    m.save()
    m2 = Manifest(path)
    assert m2.is_done("a.jpg", "depth")
    assert m2.task_status("a.jpg", "detect") == "failed"


def test_run_image_tasks_resume_and_failure(tmp_path):
    calls = []

    def runner(p):
        calls.append(p)
        return {"n_objects": 3}

    def bad_runner(p):
        raise ValueError("kaput")

    m = Manifest(tmp_path / "m.jsonl")
    report = run_image_tasks(
        tmp_path / "a.jpg", ["depth", "detect"],
        depth_runner=runner, detect_runner=bad_runner,
        manifest=m, resume=True,
    )
    assert report.processed == 1 and report.failed == 1
    m.save()

    # resume: depth skipped, detect retried
    fixed_runner = dict(depth_runner=runner, detect_runner=runner)
    report2 = run_image_tasks(
        tmp_path / "a.jpg", ["depth", "detect"],
        depth_runner=fixed_runner["depth_runner"],
        detect_runner=fixed_runner["detect_runner"],
        manifest=m, resume=True,
    )
    assert report2.skipped == 1          # depth skipped
    assert report2.processed == 1        # detect now passes
    assert m.get(str(tmp_path / "a.jpg"))["total_objects"] == 3


# -- config -----------------------------------------------------------------
def test_load_run_config_resolves_relative_paths(tmp_path):
    cfg_file = tmp_path / "run.yaml"
    cfg_file.write_text(
        "dataset:\n  image_dir: data/raw\n  out_dir: out\n"
        "tasks: [depth]\n"
        "detect:\n  ontology: ontology.yaml\n",
        encoding="utf-8",
    )
    cfg = load_run_config(cfg_file)
    assert cfg.dataset.image_dir == tmp_path / "data" / "raw"
    assert cfg.dataset.out_dir == tmp_path / "out"
    assert cfg.detect.ontology == tmp_path / "ontology.yaml"


def test_config_rejects_unknown_task():
    import yaml as _yaml
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        RunConfig.model_validate(
            _yaml.safe_load("dataset: {image_dir: a, out_dir: b}\ntasks: [nope]")
        )
