"""Run configuration: pydantic models + YAML loading.

Every tunable lives in the run config — nothing dataset-specific is hardcoded.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

VALID_TASKS = ("depth", "detect")


class DatasetCfg(BaseModel):
    image_dir: Path
    out_dir: Path
    patterns: list[str] = Field(default_factory=lambda: ["*.jpg", "*.jpeg", "*.png", "*.bmp"])


class DepthCfg(BaseModel):
    model: str = "da3-metric-L"
    device: str = "cuda"
    half: bool = True
    batch_size: int = Field(default=4, ge=1, description="Images per GPU batch")
    prefetch: int = Field(default=4, ge=1, description="Decoded images kept ahead of the GPU")
    write_workers: int = Field(default=2, ge=1, description="Threads writing PNGs while GPU infers")
    png_compression: int = Field(default=3, ge=0, le=9, description="PNG compression (0 fastest, 9 smallest)")


class DetectClassSpec(BaseModel):
    """Per-class detect spec: SAM3 prompt + optional conf_thres override."""

    prompt: str
    conf_thres: float | None = Field(default=None, ge=0.0, le=1.0)


class DetectCfg(BaseModel):
    model: str = "sam3"
    device: str = "cuda"
    half: bool = False
    conf_thres: float = Field(default=0.5, ge=0.0, le=1.0)
    max_objects_per_prompt: int = Field(default=50, ge=1)
    nms_iou: float = Field(default=0.7, gt=0.0, le=1.0)
    seg_format: str = Field(
        default="polygon", description="COCO segmentation format: polygon | rle"
    )
    prefetch: int = Field(default=2, ge=1, description="Decoded images kept ahead of the GPU")
    write_workers: int = Field(default=2, ge=1, description="Threads writing COCO/semantic while GPU infers")
    classes: dict[str, str | DetectClassSpec] = Field(
        default_factory=dict,
        description=(
            "类别名 -> SAM3 提示词。值为字符串时用全局 conf_thres；"
            "值为 {prompt, conf_thres} 时对该类单独设阈值"
        ),
    )

    @field_validator("seg_format")
    @classmethod
    def _check_seg_format(cls, v: str) -> str:
        if v not in ("polygon", "rle"):
            raise ValueError("detect.seg_format must be 'polygon' or 'rle'")
        return v

    @property
    def class_prompts(self) -> dict[str, str]:
        """类别名 -> SAM3 提示词（两种写法归一化）。"""
        return {
            name: (v if isinstance(v, str) else v.prompt)
            for name, v in self.classes.items()
        }

    @property
    def class_conf_overrides(self) -> dict[str, float]:
        """带单独阈值的类别 -> conf_thres（仅包含显式覆盖的类）。"""
        return {
            name: v.conf_thres
            for name, v in self.classes.items()
            if not isinstance(v, str) and v.conf_thres is not None
        }


class DownloadCfg(BaseModel):
    """Weight download source policy. auto = ModelScope first, HF fallback."""

    source: str = "auto"

    @field_validator("source")
    @classmethod
    def _check_source(cls, v: str) -> str:
        from ..hub import VALID_SOURCES

        if v not in VALID_SOURCES:
            raise ValueError(f"download.source must be one of: {', '.join(VALID_SOURCES)}")
        return v


class RunConfig(BaseModel):
    dataset: DatasetCfg
    tasks: list[str] = Field(default_factory=lambda: ["depth", "detect"])
    download: DownloadCfg = Field(default_factory=DownloadCfg)
    depth: DepthCfg = Field(default_factory=DepthCfg)
    detect: DetectCfg = Field(default_factory=DetectCfg)

    @field_validator("tasks")
    @classmethod
    def _check_tasks(cls, v: list[str]) -> list[str]:
        for t in v:
            if t not in VALID_TASKS:
                raise ValueError(f"Unknown task '{t}'. Valid: {', '.join(VALID_TASKS)}")
        return v


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = RunConfig.model_validate(data)
    # resolve relative paths against the config file location
    base = path.resolve().parent
    if not cfg.dataset.image_dir.is_absolute():
        cfg.dataset.image_dir = base / cfg.dataset.image_dir
    if not cfg.dataset.out_dir.is_absolute():
        cfg.dataset.out_dir = base / cfg.dataset.out_dir
    return cfg
