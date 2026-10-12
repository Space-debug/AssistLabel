"""per-class conf_thres：配置归一化 + writer 按类过滤。"""

from pathlib import Path

import numpy as np
import yaml

from assistlabel.core.config import RunConfig
from assistlabel.core.engine import Detection
from assistlabel.runner import make_detect_writer


def _cfg(tmp_path: Path) -> RunConfig:
    return RunConfig.model_validate({
        "dataset": {"image_dir": str(tmp_path / "raw"), "out_dir": str(tmp_path / "out")},
        "detect": {
            "conf_thres": 0.3,
            "classes": {
                "car": "car",
                "pedestrian": {"prompt": "person walking", "conf_thres": 0.6},
            },
        },
    })


def test_config_normalization():
    cfg = _cfg(Path("."))
    assert cfg.detect.class_prompts == {"car": "car", "pedestrian": "person walking"}
    assert cfg.detect.class_conf_overrides == {"pedestrian": 0.6}


def test_writer_filters_by_per_class_threshold(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "out").mkdir()
    writer = make_detect_writer(
        cfg,
        prompts_by_class=[("car", "car"), ("pedestrian", "person walking")],
        class_ids={"car": 1, "pedestrian": 2},
        conf_by_class={"pedestrian": 0.6},
    )
    h, w = 16, 16
    rgb = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), bool)
    mask[2:6, 2:6] = True
    dets = [
        Detection(label="car", score=0.4, box=(2, 2, 6, 6), mask=mask.copy()),       # 全局 0.3：保留
        Detection(label="person walking", score=0.5, box=(8, 8, 12, 12), mask=mask.copy()),  # 类阈值 0.6：丢弃
        Detection(label="person walking", score=0.7, box=(8, 2, 12, 6), mask=mask.copy()),   # 保留
    ]
    frag = writer(tmp_path / "raw" / "img.png", rgb, dets)
    cats = sorted(a["category_id"] for a in frag["annotations"])
    assert cats == [1, 2]


def test_yaml_accepts_both_forms(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(
        "dataset: {image_dir: a, out_dir: b}\n"
        "detect:\n"
        "  classes:\n"
        '    car: "car"\n'
        "    person: {prompt: pedestrian, conf_thres: 0.55}\n",
        encoding="utf-8",
    )
    from assistlabel.core.config import load_run_config

    cfg = load_run_config(p)
    assert cfg.detect.class_prompts["person"] == "pedestrian"
    assert cfg.detect.class_conf_overrides == {"person": 0.55}
