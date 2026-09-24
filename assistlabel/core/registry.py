"""Model registry: one YAML spec per model, inspired by X-AnyLabeling's model zoo.

Builtin specs live in ``assistlabel/config/models/*.yaml``; users can add more
by pointing ``ASSISTLABEL_MODELS_DIR`` (or ``--models-dir``) at their own folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BUILTIN_MODELS_DIR = Path(__file__).resolve().parent.parent / "config" / "models"

# kind values
KIND_DEPTH = "depth"
KIND_DETECT_SEGMENT = "detect_segment"
KIND_VIDEO = "video"

VALID_KINDS = {KIND_DEPTH, KIND_DETECT_SEGMENT, KIND_VIDEO}


@dataclass
class ModelSpec:
    key: str
    provider: str  # engine provider id: depth_anything | sam3 | mock
    kind: str  # depth | detect_segment | video
    hf_id: str = ""  # HuggingFace repo id (also the default ModelScope id)
    ms_id: str = ""  # ModelScope repo id; empty -> same as hf_id
    domain: str = "any"  # depth: indoor|outdoor|relative; detect: any
    params: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "provider": self.provider,
            "kind": self.kind,
            "hf_id": self.hf_id,
            "ms_id": self.ms_id,
            "domain": self.domain,
            "params": self.params,
            "notes": self.notes,
        }


class ModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec]):
        self._specs = specs

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, extra_dirs: list[str | Path] | None = None) -> "ModelRegistry":
        dirs: list[Path] = [BUILTIN_MODELS_DIR]
        env_dir = os.environ.get("ASSISTLABEL_MODELS_DIR")
        if env_dir:
            dirs.append(Path(env_dir))
        for d in extra_dirs or []:
            dirs.append(Path(d))

        specs: dict[str, ModelSpec] = {}
        for d in dirs:
            if not d.is_dir():
                continue
            for path in sorted(d.glob("*.yaml")):
                for spec in _load_file(path):
                    specs[spec.key] = spec  # later dirs override builtins
        return cls(specs)

    # -- access ------------------------------------------------------------
    def get(self, key: str) -> ModelSpec:
        if key not in self._specs:
            raise KeyError(
                f"Unknown model key '{key}'. Available: {', '.join(sorted(self._specs))}"
            )
        return self._specs[key]

    def keys(self) -> list[str]:
        return sorted(self._specs)

    def all(self) -> list[ModelSpec]:
        return [self._specs[k] for k in self.keys()]

    def filter_kind(self, kind: str) -> list[ModelSpec]:
        return [s for s in self.all() if s.kind == kind]


def _load_file(path: Path) -> list[ModelSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("models", data if isinstance(data, list) else [data])
    specs: list[ModelSpec] = []
    for item in models:
        spec = ModelSpec(
            key=item["key"],
            provider=item["provider"],
            kind=item["kind"],
            hf_id=item.get("hf_id", ""),
            ms_id=item.get("ms_id", ""),
            domain=item.get("domain", "any"),
            params=item.get("params", {}) or {},
            notes=item.get("notes", ""),
        )
        if spec.kind not in VALID_KINDS:
            raise ValueError(f"{path}: model '{spec.key}' has invalid kind '{spec.kind}'")
        specs.append(spec)
    return specs
