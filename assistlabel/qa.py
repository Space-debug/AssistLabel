"""QA: dataset-level statistics, HTML report, stratified review sampling.

Stats are derived from the native ``detect/annotations.json`` COCO store.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .io.dataset import out_paths

SCORE_BUCKETS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def _coco_path(out_dir: Path) -> Path:
    return Path(out_dir) / "detect" / "annotations.json"


def collect_stats(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    coco_path = _coco_path(out_dir)
    if not coco_path.exists():
        return {
            "images": 0, "class_counts": {}, "score_hist": {},
            "objects_per_image": {"mean": 0, "max": 0, "min": 0},
            "depth_valid_ratio_mean": None, "errors": [],
        }
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}

    class_counts: Counter = Counter()
    score_hist: Counter = Counter()
    per_image: list[int] = []
    for ann in coco.get("annotations", []):
        class_counts[cats.get(ann.get("category_id"), "?")] += 1
        score = float(ann.get("score", 1.0))
        for i, edge in enumerate(SCORE_BUCKETS):
            if score < edge:
                low = SCORE_BUCKETS[i - 1] if i else 0.0
                score_hist[f"{low:.1f}-{edge:.1f}"] += 1
                break
    per_image = Counter(a["image_id"] for a in coco.get("annotations", [])).values()
    counts = list(per_image)

    # depth validity: sampled from the depth PNGs (capped for big sets)
    ratios = _sample_depth_valid_ratios(out_dir, limit=200)

    return {
        "images": len(coco.get("images", [])),
        "class_counts": dict(class_counts.most_common()),
        "score_hist": dict(score_hist),
        "objects_per_image": {
            "mean": round(sum(counts) / len(counts), 2) if counts else 0,
            "max": max(counts) if counts else 0,
            "min": min(counts) if counts else 0,
        },
        "depth_valid_ratio_mean": (
            round(sum(ratios) / len(ratios), 4) if ratios else None
        ),
        "errors": [],
    }


def _sample_depth_valid_ratios(out_dir: Path, limit: int = 200) -> list[float]:
    ratios = []
    depth_dir = out_dir / "depth"
    if not depth_dir.is_dir():
        return ratios
    for p in sorted(depth_dir.glob("*.png"))[:limit]:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            continue
        ratios.append(round(float((raw > 0).mean()), 4))
    return ratios


def write_report(stats: dict, path: str | Path) -> None:
    import html

    obj = stats["objects_per_image"]
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in stats["class_counts"].items()
    )
    score_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in stats["score_hist"].items()
    )
    error_rows = "".join(f"<li>{html.escape(e)}</li>" for e in stats["errors"]) or "<li>none</li>"
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>AssistLabel QA report</title>
<style>
body{{font-family:Segoe UI,Helvetica,sans-serif;margin:2rem;color:#222}}
h1{{font-size:1.4rem}} table{{border-collapse:collapse;margin:1rem 0}}
td,th{{border:1px solid #ccc;padding:4px 12px}} th{{background:#f5f5f5}}
.kpi{{display:inline-block;margin-right:2rem;padding:1rem 1.4rem;background:#f0f6ff;border-radius:8px}}
.kpi b{{font-size:1.6rem;display:block}}
</style></head><body>
<h1>AssistLabel QA report</h1>
<div class="kpi"><b>{stats['images']}</b>labeled images</div>
<div class="kpi"><b>{obj['mean']}</b>objects/image (mean)</div>
<div class="kpi"><b>{obj['max']}</b>max objects/image</div>
<div class="kpi"><b>{stats.get('depth_valid_ratio_mean')}</b>mean depth valid ratio</div>
<h2>Class distribution</h2><table><tr><th>class</th><th>count</th></tr>{rows}</table>
<h2>Confidence histogram</h2><table><tr><th>score</th><th>count</th></tr>{score_rows}</table>
<h2>Per-file errors</h2><ul>{error_rows}</ul>
</body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def sample_review_list(out_dir: str | Path, n: int = 50, seed: int = 0) -> list[str]:
    """Stratified-by-confidence review list: images ranked by their lowest
    annotation score; the weakest n are returned."""
    out_dir = Path(out_dir)
    coco_path = _coco_path(out_dir)
    if not coco_path.exists():
        return []
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    files = {im["id"]: im["file_name"] for im in coco.get("images", [])}
    min_score: dict[int, float] = {}
    for ann in coco.get("annotations", []):
        iid = ann.get("image_id")
        s = float(ann.get("score", 1.0))
        min_score[iid] = min(min_score.get(iid, 1.01), s)
    ranked = sorted(min_score.items(), key=lambda kv: kv[1])
    picked = [files[iid] for iid, _ in ranked[:n]]
    return picked
