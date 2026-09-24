"""AssistLabel CLI — 批量自动标注：Depth Anything 深度真值 + SAM3 检测/实例分割。

所有命令与子命令都支持 --help 查询用法（如 assistlabel run --help），
也可用 assistlabel help <命令> 查看。
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

detect:
  model: "{detect_model}"
  ontology: "ontology.yaml"
  device: cuda
  half: false
  conf_thres: 0.5
  max_objects_per_prompt: 50
  nms_iou: 0.7
  prefetch: 2                  # decoded images kept ready ahead of the GPU
  write_workers: 2             # COCO/semantic write threads overlapping inference

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


@app.command(help="生成项目配置：run.yaml + ontology.yaml + data/raw 目录")
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


@app.command(help="批量标注流水线：断点续跑、崩溃安全；depth+detect 串行执行")
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


@app.command(help="查看各任务进度（done/failed/pending/ETA），只读不执行")
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




@app.command(help="导出标注：yolo=ultralytics 布局（semantic/depth 由 run 原生产出）")
def export(
    config: Path = typer.Option(Path("run.yaml"), "--config", "-c"),
    fmt: str = typer.Option("yolo", "--format", help="yolo（ultralytics 训练布局）"),
    split: float = typer.Option(0.75, "--split", help="Train fraction (<1.0 writes a val split too)."),
):
    """从 detect 的 COCO 标注导出 ultralytics 训练布局。

    detect/annotations.json 本身就是标准 COCO，可直接用 pycocotools 读取；
    yolo 布局按需从此派生。
    """
    from .io.annotations import export_ultralytics
    from .io.dataset import scan_images

    cfg = _load_cfg(config)
    if fmt != "yolo":
        err_console.print(f"Unknown format: {fmt}（semantic/depth 由 run 原生产出，无需导出）")
        raise typer.Exit(code=2)

    coco_json = Path(cfg.dataset.out_dir) / "detect" / "annotations.json"
    if not coco_json.exists():
        err_console.print(f"未找到 {coco_json}，请先运行 detect 任务")
        raise typer.Exit(code=1)

    out_dir = Path(cfg.dataset.out_dir) / "ultralytics"
    stats = export_ultralytics(
        coco_json, out_dir, image_dir=cfg.dataset.image_dir, split=split,
    )
    console.print(
        f"YOLO -> [cyan]{out_dir}[/]: train={stats['train']} val={stats['val']} "
        "(images/ labels/ data.yaml)"
    )


@app.command("viz")
def viz_cmd(
    config: Path = typer.Option(Path("run.yaml"), "--config", "-c"),
    kind: str = typer.Option("depth,detect,semantic", "--kind",
                             help="逗号分隔: depth,detect,semantic"),
):
    """从已有标注产物生成/刷新可视化（depth_viz/detect_viz/semantic_viz），不重新推理。"""
    from .runner import build_viz

    cfg = _load_cfg(config)
    counts = build_viz(cfg, [k.strip() for k in kind.split(",")])
    console.print(
        f"可视化已生成: depth_viz={counts['depth']} 张, detect_viz={counts['detect']} 张, "
        f"semantic_viz={counts['semantic']} 张"
    )


@app.command(help="QA 质检：类别分布/置信度/深度有效率报告 + 复核清单")
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


@app.command(help="产物完整性校验：撕裂写/源图变更检测，--repair 重置坏条目")
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


@app.command("help")
def help_cmd(
    command: list[str] = typer.Argument(
        None, help="要查询的命令名，支持多级：run / models / models download"
    ),
):
    """显示命令的使用说明（等同 --help），支持多级子命令。"""
    import click

    import typer.main

    root = typer.main.get_command(app)
    ctx = click.Context(root, info_name="assistlabel")
    target = root
    for part in command or []:
        # 鸭子类型判断 Group（部分环境下 isinstance 检查不可靠）
        if not hasattr(target, "get_command"):
            err_console.print(f"'{' '.join(command)}' 没有子命令 '{part}'")
            raise typer.Exit(code=2)
        sub = target.get_command(ctx, part)
        if sub is None:
            err_console.print(f"未知命令 '{part}'（assistlabel --help 查看全部命令）")
            raise typer.Exit(code=2)
        ctx = click.Context(sub, parent=ctx, info_name=part)
        target = sub
    console.print(target.get_help(ctx))


# ---------------------------------------------------------------------------
@models_app.command("list")
def models_list(
    kind: str = typer.Option(None, "--kind", help="按类型过滤: depth | detect_segment"),
):
    """浏览所有已注册模型（含中文说明，挑选后再下载）。"""
    registry = ModelRegistry.load()
    models = [m for m in registry.all()
              if kind is None or m.kind == kind or m.provider == kind]
    table = Table(title="Model registry — 挑选后执行 assistlabel models download <key>")
    for col, w in (("key", None), ("kind", None), ("适用", None), ("权重仓库", None), ("说明", 46)):
        table.add_column(col, overflow="fold" if w else None, max_width=w)
    kind_cn = {"depth": "深度", "detect_segment": "检测+实例分割", "video": "视频"}
    for spec in models:
        domain_cn = {"metric": "米制深度", "relative": "相对深度",
                     "indoor": "室内", "outdoor": "室外", "any": "通用"}.get(spec.domain, spec.domain)
        table.add_row(
            f"[bold]{spec.key}[/]", kind_cn.get(spec.kind, spec.kind),
            domain_cn, spec.hf_id or "-",
            (spec.notes or "-")[:56],
        )
    console.print(table)
    console.print("下载: assistlabel models download <key>   详情: assistlabel models info <key>")


@models_app.command("info")
def models_info(key: str = typer.Argument(..., help="模型 key，见 models list")):
    """查看单个模型的完整规格。"""
    registry = ModelRegistry.load()
    try:
        spec = registry.get(key)
    except KeyError as e:
        err_console.print(str(e))
        raise typer.Exit(code=2) from e
    console.print_json(json.dumps(spec.to_dict(), ensure_ascii=False))


@models_app.command("download")
def models_download(
    key: str = typer.Argument(..., help="模型 key，见 models list"),
    source: str = typer.Option("auto", "--source", help="auto | modelscope | huggingface"),
):
    """预下载模型权重（默认 ModelScope 优先，国内直连）。"""
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
    """环境体检：torch/CUDA、transformers、modelscope、下载源逐项检查。"""
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
