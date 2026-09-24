"""Ontology: dataset class name <-> natural-language concept prompt mapping
(autodistill-style). Keeps the label taxonomy decoupled from SAM3 prompts.

YAML format::

    classes:
      car:            {prompt: "car"}
      pedestrian:     {prompt: "pedestrian walking on the road"}

Shorthand ``car: "car"`` is also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

EXAMPLE = """\
# AssistLabel ontology: dataset class name -> SAM3 concept prompt.
# The label written into annotations is the key; the prompt is sent to SAM3.
classes:
  car:            {prompt: "car"}
  pedestrian:     {prompt: "pedestrian"}
  traffic_cone:   {prompt: "traffic cone"}
  truck:          {prompt: "truck"}
"""


class _NoDuplicateKeysLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently
    keeping the last one (a duplicated class name is almost always a typo)."""


def _no_dup_constructor(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key in ontology: '{key}'")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicateKeysLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup_constructor
)


@dataclass
class ClassDef:
    name: str  # label written into annotations
    prompt: str  # text concept prompt for SAM3


def load_ontology(path: str | Path) -> list[ClassDef]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_NoDuplicateKeysLoader) or {}

    classes = data.get("classes")
    if not classes or not isinstance(classes, dict):
        raise ValueError(f"{path}: expected top-level 'classes:' mapping")

    out: list[ClassDef] = []
    seen: set[str] = set()
    for name, value in classes.items():
        name = str(name).strip()
        if not name:
            raise ValueError(f"{path}: empty class name")
        if name in seen:
            raise ValueError(f"{path}: duplicate class '{name}'")
        seen.add(name)
        if isinstance(value, str):
            prompt = value
        elif isinstance(value, dict):
            prompt = str(value.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"{path}: class '{name}' has empty prompt")
        else:
            raise ValueError(f"{path}: class '{name}' must be a string or a mapping")
        out.append(ClassDef(name=name, prompt=prompt))

    if not out:
        raise ValueError(f"{path}: ontology defines no classes")
    return out
