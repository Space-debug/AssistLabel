"""AssistLabel CLI.

Commands:
  init      Scaffold run.yaml + ontology.yaml
  run       batch pipeline (depth / detect, resumable)
  export    labelme -> COCO / YOLO
  validate  QA report + stratified review list
  models    registry inspection + environment preflight
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .core.ontology import EXAMPLE as ONTOLOGY_EXAMPLE
from .core.registry import ModelRegistry
from .io.dataset import out_paths, scan_images

app = typer.Typer(
    name="assistlabel",
    help="Batch auto-labeling: Depth Anything ground truth + SAM3 detection/segmentation.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Model registry inspection and environment checks.", no_args_is_help=True)
app.add_typer(models_app, name="models")

console = Console()
err_console = Console(stderr=True, style="bold red")

RUN_YAML_TEMPLATE = """\
# AssistLabel run config. Paths are resolved relative to this file.
dataset:
  image_dir: "data/raw"
  out_dir: "data/labeled"
  patterns: ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

tasks: [depth, detect]          # any of: depth / detect

download:
  source: auto                  # auto = ModelScope 优先, HF 兜底; 或 modelscope / huggingface

depth:
  model: "{depth_model}"
  device: cuda
  half: true
  batch_size: 4                # GPU batch; raise on big VRAM (8-16), lower on OOM
  prefetch: 4                  # decoded images kept ready ahead of the GPU
  write_workers: 2             # PNG/meta write threads overlapping inference
  png_compression: 3           # 0 fastest / 9 smallest
  save_raw: true               # 16-bit PNG, millimeters
  save_color: true             # pseudo-color JPG for eyeballing
  save_npz: false

detect:
  model: "{detect_model}"
  ontology: "ontology.yaml"
  device: cuda
  half: false
  conf_thres: 0.5
  max_objects_per_prompt: 50
  nms_iou: 0.7
  save_viz: true
  prefetch: 2                  # decoded images kept ready ahead of the GPU
  write_workers: 2             # labelme/viz write threads overlapping inference

"""


def _load_cfg(config: Path):
    from .core.config import load_run_config

    try:
        return load_run_config(config)
    except FileNotFoundError as e:
        err_console.print(str(e))
        raise typer.Exit(code=2) from e
    except Exception as e:
        err_console.print(f"Invalid config: {e}")
        raise typer.Exit(code=2) from e


def _report(report):
    from .core.pipeline import RunReport  # noqa: F401 (typing only)

    color = "green" if report.failed == 0 else "yellow"
    throughput = f" ({report.processed / report.seconds:.1f} img/s)" if report.seconds > 0 and report.processed else ""
    console.print(
        f"[{color}]done[/] processed={report.processed} skipped(resume)={report.skipped} "
        f"failed={report.failed} elapsed={report.seconds:.1f}s{throughput}"
    )
    for failure in report.failures:
        err_console.print(f"  FAILED {failure['task']}: {failure['image']} -> {failure['error']}")


# ---------------------------------------------------------------------------
# preferred default models for `assistlabel init --engine auto` (registry keys
# are iterated in sorted order otherwise, which would pick mock/beta variants)
_DEPTH_PREFERENCE = [
    "da3-metric-L",
    "da2-metric-indoor-L", "da2-metric-outdoor-L",
    "da2-metric-indoor-B", "da2-metric-outdoor-B", "da2-relative-L",
]
_DETECT_PREFERENCE = ["sam3"]

# providers whose engines need explicit file downloads (hub.resolve_model_dir_files)
_FILE_BASED_PROVIDERS = {
    "sam3": "assistlabel.engines.sam3",
    "depth_anything3": "assistlabel.engines.depth_anything3",
}


def _pick_default(registry: ModelRegistry, kind: str, preference: list[str]) -> str | None:
    models = {s.key: s for s in registry.filter_kind(kind) if not s.provider.startswith("mock")}
    for key in preference:
        if key in models:
            return key
    return next(iter(models), None)


@app.command()
def init(
    dir: Path = typer.Option(Path("."), "--dir", help="Project directory to scaffold."),
    engine: str = typer.Option("auto", "--engine", help="auto | mock (mock = no GPU deps, for pipeline testing)."),
):
    """Scaffold run.yaml + ontology.yaml for a new dataset project."""
    (dir / "data" / "raw").mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry.load()
    if engine == "mock":
        depth_model, detect_model = "mock-depth", "mock-detect"
    else:
        depth_model = _pick_default(registry, "depth", _DEPTH_PREFERENCE) or "da2-metric-indoor-L"
        detect_model = _pick_default(registry, "detect_segment", _DETECT_PREFERENCE) or "sam3"

    run_yaml = dir / "run.yaml"
    run_yaml.write_text(
        RUN_YAML_TEMPLATE.format(depth_model=depth_model, detect_model=detect_model),
        encoding="utf-8",
    )
    (dir / "ontology.yaml").write_text(ONTOLOGY_EXAMPLE, encoding="utf-8")
    console.print(f"Created [cyan]{run_yaml}[/], [cyan]{dir / 'ontology.yaml'}[/]")
    console.print(f"  depth model:  {depth_model}\n  detect model: {detect_model}")
    console.print("Next: drop images into data/raw, edit ontology.yaml, then:\n"
                  "  assistlabel run -c run.yaml")


@app.command()
def run(
    config: Path = typer.Option(Path("run.yaml"), "--config", "-c"),
    tasks: str = typer.Option(None, "--tasks", help="Comma list overriding config tasks, e.g. depth,detect."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip images already done in manifest."),
    limit: int = typer.Option(None, "--limit", help="Only process the first N images (pilot runs)."),
    redo_detect: bool = typer.Option(False, "--redo-detect", help="Clear all detect statuses first (e.g. after ontology changes)."),
    force: bool = typer.Option(False, "--force", help="Skip the free-disk-space precheck."),
):
    """Run the batch labeling pipeline (resumable, crash-safe)."""
    from .runner import run_pipeline

    cfg = _load_cfg(config)
    task_list = [t.strip() for t in tasks.split(",")] if tasks else cfg.tasks
    redo = ["detect"] if redo_detect else []
    try:
        report = run_pipeline(cfg, task_list, resume=resume, limit=limit,
                              redo_tasks=redo, force=force)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted - manifest saved, rerun with --resume to continue.[/]")
        raise typer.Exit(code=130) from None
    except ManifestLockedError as e:
        err_console.print(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        err_console.print(f"Pipeline error: {e}")
        raise typer.Exit(code=1) from e
    _report(report)
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def status(
    config: Path = typer.Option(Path("run.yaml"), "--config", "-c"),
    tasks: str = typer.Option(None, "--tasks", help="Comma list of tasks to report (default: config tasks)."),
):
    """Show per-task progress (done / failed / pending / ETA) without running anything."""
    from .core.ontology import load_ontology
    from .core.pipeline import Manifest
    from .io.dataset import out_paths, scan_images

    cfg = _load_cfg(config)
    images = scan_images(cfg.dataset.image_dir, cfg.dataset.patterns)
    manifest = Manifest(out_paths(cfg.dataset.out_dir, "x")["manifest"])
    task_list = [t.strip() for t in tasks.split(",")] if tasks else cfg.tasks

    if detect_config_changed(cfg, manifest):
        console.print(
            "[yellow]ontology.yaml changed since these labels were produced.\n"
            "  Re-run with --redo-detect to refresh all detect labels.[/]"
        )

    table = Table(title=f"Progress ({len(images)} images discovered)")
    for col in ("task", "done", "failed", "pending", "avg s/img", "eta"):
        table.add_column(col)
    import time as _time

    for task in task_list:
        done = failed = 0
        seconds = []
        for rec in manifest.records.values():
            entry = rec.get("tasks", {}).get(task, {})
            if entry.get("status") == "done":
                done += 1
                if entry.get("seconds") is not None:
                    seconds.append(float(entry["seconds"]))
            elif entry.get("status") == "failed":
                failed += 1
        pending = max(len(images) - done - failed, 0)
        avg = sum(seconds) / len(seconds) if seconds else None
        eta = _fmt_eta(avg * pending) if (avg is not None and pending) else "-"
        table.add_row(task, str(done), str(failed) if failed else "0", str(pending),
                      f"{avg:.2f}" if avg is not None else "-", eta)
    console.print(table)


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}min"
    return f"{seconds / 3600:.1f}h"


def detect_config_changed(cfg, manifest) -> bool:
    """True when the ontology hash on disk differs from stored detect labels."""
    import hashlib

    if not cfg.detect.ontology.exists():
        return False
    current = hashlib.sha1(cfg.detect.ontology.read_bytes()).hexdigest()[:12]
    mismatched = sum(
        1 for rec in manifest.records.values()
        if rec.get("tasks", {}).get("detect", {}).get("status") == "done"
        and rec["tasks"]["detect"].get("ontology_hash") not in (None, current)
    )
    return mismatched > 0




@app.command()
def export(
    config: Path = typer.Option(Path("run.yaml"), "--config", "-c"),
    fmt: str = typer.Option("coco", "--format", help="coco | yolo | semantic"),
    split: float = typer.Option(1.0, "--split", help="Train fraction (<1.0 writes a val split too)."),
    segmentation: str = typer.Option("polygon", "--segmentation", help="COCO: polygon | rle"),
    viz: bool = typer.Option(False, "--viz", help="semantic: also write color-coded preview PNGs"),
):
    """Export labelme labels to COCO / YOLO (ultralytics) / semantic mask format."""
    from .io.annotations import export_coco, export_semantic, export_ultralytics
    from .core.ontology import load_ontology

    cfg = _load_cfg(config)
    classes = [c.name for c in load_ontology(cfg.detect.ontology)]
    paths = out_paths(cfg.dataset.out_dir, "x")  # only need dirs
    labels_dir = Path(paths["labelme"]).parent

    if fmt == "coco":
        out_json = Path(paths["coco"])
        coco = export_coco(labels_dir, out_json, classes, segmentation=segmentation,
                           split=split, image_dir=cfg.dataset.image_dir)
        n_val = ""
        if split < 1.0:
            n_val = f" (+val: {out_json.parent / 'annotations_val.json'})"
        console.print(
            f"COCO -> [cyan]{out_json}[/]: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations{n_val}"
        )
    elif fmt == "yolo":
        from .io.dataset import scan_images

        out_dir = Path(cfg.dataset.out_dir) / "ultralytics"
        stats = export_ultralytics(
            labels_dir, out_dir, classes,
            image_files=scan_images(cfg.dataset.image_dir, cfg.dataset.patterns),
            split=split,
        )
        console.print(
            f"YOLO -> [cyan]{out_dir}[/]: train={stats['train']} val={stats['val']} "
            "(images/ labels/ data.yaml)"
        )
    elif fmt == "semantic":
        out_dir = Path(cfg.dataset.out_dir) / "semantic"
        n = export_semantic(labels_dir, out_dir, classes, viz=viz)
        console.print(
            f"Semantic -> [cyan]{out_dir}[/]: {n} class-index PNGs "
            "(0=background, i=ontology order) (+classes.txt)"
            + (f" + 彩色预览 {out_dir.parent / 'semantic_viz'}" if viz else "")
        )
    else:
        err_console.print(f"Unknown format: {fmt}")
        raise typer.Exit(code=2)


@app.command()
def validate(
    config: Path = typer.Option(Path("run.yaml"), "--config", "-c"),
    report_path: Path = typer.Option(None, "--report", help="HTML report output path."),
    sample: int = typer.Option(50, "--sample", help="Size of the stratified review list (0 disables)."),
):
    """QA statistics, HTML report, and a lowest-confidence review list."""
    from .qa import collect_stats, sample_review_list, write_report

    cfg = _load_cfg(config)
    stats = collect_stats(cfg.dataset.out_dir)
    report_path = report_path or (Path(cfg.dataset.out_dir) / "report.html")
    write_report(stats, report_path)
    console.print(f"Report -> [cyan]{report_path}[/]")

    table = Table(title="Class distribution")
    table.add_column("class"); table.add_column("count", justify="right")
    for k, v in stats["class_counts"].items():
        table.add_row(k, str(v))
    console.print(table)
    if stats["depth_valid_ratio_mean"] is not None:
        console.print(f"Depth valid-pixel ratio (mean): {stats['depth_valid_ratio_mean']}")

    if sample > 0:
        review = sample_review_list(cfg.dataset.out_dir, n=sample)
        list_path = Path(cfg.dataset.out_dir) / "review_list.txt"
        list_path.write_text("\n".join(review), encoding="utf-8")
        console.print(f"Review list ({len(review)} images, lowest-confidence first) -> [cyan]{list_path}[/]")


@app.command()
def verify(
    config: Path = typer.Option(Path("run.yaml"), "--config", "-c"),
    repair: bool = typer.Option(False, "--repair", help="Reset broken manifest entries for reprocessing."),
):
    """Check every 'done' record's files are intact (crash / torn-write detection)."""
    from .verify import Issue, repair as do_repair, verify_outputs
    from rich.table import Table as RichTable

    cfg = _load_cfg(config)
    issues = verify_outputs(cfg)
    if not issues:
        console.print("[green]All outputs intact.[/]")
        return
    table = RichTable(title=f"{len(issues)} integrity issue(s)")
    for col in ("image", "task", "kind", "detail"):
        table.add_column(col)
    for issue in issues[:30]:
        table.add_row(Path(issue.image).name, issue.task, issue.kind, issue.detail)
    console.print(table)
    if len(issues) > 30:
        console.print(f"  ... and {len(issues) - 30} more")
    if repair:
        n = do_repair(cfg, issues)
        console.print(f"[yellow]Reset {n} manifest entries.[/] Re-run with --resume to reprocess them.")
    else:
        console.print("Run again with --repair to reset these entries for reprocessing.")


# ---------------------------------------------------------------------------
@models_app.command("list")
def models_list():
    """List all registered models."""
    registry = ModelRegistry.load()
    table = Table(title="Model registry")
    for col in ("key", "provider", "kind", "domain", "hf_id"):
        table.add_column(col)
    for spec in registry.all():
        table.add_row(spec.key, spec.provider, spec.kind, spec.domain, spec.hf_id)
    console.print(table)


@models_app.command("info")
def models_info(key: str = typer.Argument(...)):
    """Show one model's full spec."""
    registry = ModelRegistry.load()
    try:
        spec = registry.get(key)
    except KeyError as e:
        err_console.print(str(e))
        raise typer.Exit(code=2) from e
    console.print_json(json.dumps(spec.to_dict(), ensure_ascii=False))


@models_app.command("download")
def models_download(
    key: str = typer.Argument(...),
    source: str = typer.Option("auto", "--source", help="auto | modelscope | huggingface"),
):
    """Pre-download weights for a registry model (ModelScope first by default)."""
    from . import hub
    from .core.registry import KIND_DEPTH, ModelRegistry

    if source not in hub.VALID_SOURCES:
        err_console.print(f"Unknown source '{source}'. Valid: {', '.join(hub.VALID_SOURCES)}")
        raise typer.Exit(code=2)
    try:
        spec = ModelRegistry.load().get(key)
    except KeyError as e:
        err_console.print(str(e))
        raise typer.Exit(code=2) from e
    if spec.provider.startswith("mock"):
        err_console.print(f"'{key}' is a mock model - nothing to download.")
        raise typer.Exit(code=2)

    with console.status(f"[bold]Downloading {key} (source={source}) ..."):
        if spec.provider in _FILE_BASED_PROVIDERS:
            import importlib

            mod = importlib.import_module(_FILE_BASED_PROVIDERS[spec.provider])
            path = hub.resolve_model_dir_files(
                spec.hf_id, spec.ms_id, mod.REQUIRED_FILES, source=source
            )
            if path is None:
                err_console.print(
                    f"Could not fetch {key}: install 'modelscope' or 'huggingface_hub' "
                    "and check network / token access."
                )
                raise typer.Exit(code=1)
            console.print(f"Weights ready -> [cyan]{path}[/]")
        else:
            # snapshot-based providers (depth engines)
            path = hub.resolve_model_dir(spec.hf_id, spec.ms_id, source=source)
            console.print(f"Model ready -> [cyan]{path}[/]")


@models_app.command("check")
def models_check():
    """Environment preflight: torch/CUDA, transformers, sam3, HF token, weights."""
    import importlib.util
    import os

    rows: list[tuple[str, str, str]] = []

    def check(name: str, module: str, hint: str = ""):
        found = importlib.util.find_spec(module) is not None
        detail = "installed" if found else f"missing  {hint}"
        rows.append((name, "ok" if found else "fail", detail))

    torch_ok = importlib.util.find_spec("torch") is not None
    if torch_ok:
        import torch

        rows.append(("torch", "ok", f"v{torch.__version__}, cuda={torch.cuda.is_available()}"))
    else:
        rows.append(("torch", "fail", "missing  pip install torch --index-url https://download.pytorch.org/whl/cu128"))
    check("transformers", "transformers", "pip install 'transformers>=5'  # SAM3 support")
    check("modelscope (download source)", "modelscope",
          "pip install modelscope  # ModelScope 优先下载源, 国内直连")
    check("pycocotools (optional)", "pycocotools", "pip install pycocotools  # COCO RLE export")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    rows.append(("HF_TOKEN", "ok" if token else "warn",
                 "set (HF fallback source)" if token else
                 "not set - only needed when falling back to huggingface.co"))
    rows.append(("Download source", "ok",
                 "run.yaml download.source=auto -> ModelScope 优先, HF 兜底"))

    table = Table(title="Environment preflight")
    table.add_column("component")
    table.add_column("status")
    table.add_column("detail")
    for name, status, detail in rows:
        style = {"ok": "green", "warn": "yellow", "fail": "red"}.get(status, "")
        table.add_row(name, f"[{style}]{status}[/]", detail)
    console.print(table)


def _version_callback(value: bool):
    if value:
        console.print(f"assistlabel {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(False, "--version", "-V", callback=_version_callback, is_eager=True),
):
    pass


if __name__ == "__main__":
    app()
