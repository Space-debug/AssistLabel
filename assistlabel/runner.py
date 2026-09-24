"""Pipeline orchestrator: wires config + registry + engines + manifest.

Phases run serially (depth -> detect) so one GPU serves all engines.

Outputs (per the user-facing layout):

- ``detect/annotations.json``  COCO instance seg+det (RLE masks, bbox, score,
  per-object depth stats)
- ``semantic/<stem>.png``      class-index masks (0=background, i=ontology order)
- ``depth/<stem>.png``         16-bit millimeters (KITTI convention)

The depth phase is a three-stage pipeline for maximum throughput on large
datasets: a producer thread decodes images ahead of the GPU (``prefetch``),
the GPU runs true batched inference (``batch_size``, orientation-bucketed),
and a thread pool writes PNGs while the GPU infers the next batch. The
detect phase mirrors this: decode -> (image, prompt) pair inference -> COCO /
semantic / viz writes. Batch-level failures (OOM) degrade to per-image so one
bad image never poisons its batch.
"""

from __future__ import annotations

import hashlib
import queue
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .core.config import RunConfig
from .core.engine import BaseEngine, Detection, EngineError
from .core.pipeline import STATUS_DONE, STATUS_FAILED, Manifest, RunReport, run_image_tasks
from .core.registry import ModelRegistry
from .engines.mock import MockDepthEngine, MockDetectEngine
from .fusion.box_depth import annotate_detections_with_depth
from .io.annotations import (
    coco_load,
    coco_next_annotation_id,
    coco_save,
    coco_upsert_image,
    rle_encode,
)
from .io.dataset import file_hash, out_paths, scan_images
from .io.depth_io import read_depth_png, save_depth_color, save_depth_png
from .viz.overlay import draw_detections

console = Console()

_ENGINE_CACHE: dict[str, BaseEngine] = {}


def build_engine(provider: str, hf_id: str, params: dict) -> BaseEngine:
    if provider in _ENGINE_CACHE:
        return _ENGINE_CACHE[provider]
    if provider == "depth_anything":
        engine = _import("DepthAnythingEngine", "assistlabel.engines.depth_anything")(hf_id, params)
    elif provider == "depth_anything3":
        engine = _import("DepthAnything3Engine", "assistlabel.engines.depth_anything3")(hf_id, params)
    elif provider == "sam3":
        engine = _import("Sam3Engine", "assistlabel.engines.sam3")(hf_id, params)
    elif provider == "mock_depth":
        engine = MockDepthEngine(params)
    elif provider == "mock_detect":
        engine = MockDetectEngine(params)
    else:
        raise EngineError(f"Unknown engine provider '{provider}'")
    _ENGINE_CACHE[provider] = engine
    return engine


def _import(symbol: str, module: str):
    import importlib

    return getattr(importlib.import_module(module), symbol)


def _read_image_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)  # windows-safe unicode paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _write_image_jpg(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img_bgr)
    if ok:
        buf.tofile(str(path))


def _is_portrait(img: np.ndarray) -> bool:
    return img.shape[0] > img.shape[1]


def _relative_name(image_path: Path, image_dir: Path | None) -> str:
    """Portable COCO file_name: relative to image_dir, forward slashes."""
    import os

    if image_dir is not None:
        try:
            return os.path.relpath(image_path, image_dir).replace("\\", "/")
        except ValueError:
            pass
    return image_path.name


def _engine_params(spec) -> dict:
    """Propagate the registry ms_id; engines read params["ms_id"]."""
    return {**spec.params, "ms_id": spec.ms_id} if spec.ms_id else dict(spec.params)


# ---------------------------------------------------------------------------
# depth phase: pipelined (prefetch decode | batched GPU | async writes)
# ---------------------------------------------------------------------------
def make_depth_writer(cfg: RunConfig):
    """fn(image_path, result) -> manifest extras; writes all depth outputs."""

    def write(image_path: Path, result: dict) -> dict:
        paths = out_paths(cfg.dataset.out_dir, image_path, cfg.dataset.image_dir)
        depth: np.ndarray = result["depth_m"]

        # sanity sentinel: near-empty or absurd depth means the engine or
        # preprocessing is misaligned — fail the image instead of storing junk
        invalid_ratio = float((~np.isfinite(depth) | (depth <= 0)).mean())
        if invalid_ratio > 0.95:
            raise ValueError(
                f"depth map has {invalid_ratio:.0%} invalid pixels (engine/preprocessing misaligned?)"
            )
        dmax = float(np.nanmax(depth)) if np.isfinite(depth).any() else 0.0
        if dmax > 200.0:
            raise ValueError(f"depth max {dmax:.1f}m exceeds 200m sanity bound")

        save_depth_png(depth, paths["depth_png"], compression=cfg.depth.png_compression)
        if cfg.depth.viz:
            save_depth_color(depth, paths["depth_viz"])
        return {
            "source_image_hash": file_hash(image_path),
            "model": cfg.depth.model,
        }

    return write


def _depth_write_worker(write, manifest: Manifest, image_path: Path, result: dict):
    """Runs inside the write pool: write outputs, then mark the manifest."""
    try:
        extra = write(image_path, result)
        manifest.mark(str(image_path), "depth", STATUS_DONE, **extra)
        return str(image_path), True, None
    except Exception as exc:  # noqa: BLE001
        manifest.mark(str(image_path), "depth", STATUS_FAILED, error=str(exc))
        return str(image_path), False, str(exc)


def _run_depth_phase(
    cfg: RunConfig,
    registry: ModelRegistry,
    images: list[Path],
    manifest: Manifest,
    report: RunReport,
    progress: Progress,
    pid,
) -> None:
    spec = registry.get(cfg.depth.model)
    engine = build_engine(spec.provider, spec.hf_id, _engine_params(spec))
    engine.load(device=cfg.depth.device, half=cfg.depth.half, source=cfg.download.source)
    write = make_depth_writer(cfg)
    batch_size = max(1, cfg.depth.batch_size)

    # stage 1: producer thread keeps decoded images ready for the GPU
    q: queue.Queue = queue.Queue(maxsize=max(2, cfg.depth.prefetch))
    sentinel = object()

    def produce() -> None:
        for p in images:
            try:
                q.put((p, _read_image_rgb(p), None))
            except Exception as exc:  # corrupt / undecodable file
                q.put((p, None, exc))
        q.put(sentinel)

    threading.Thread(target=produce, daemon=True).start()

    write_pool = ThreadPoolExecutor(max_workers=max(1, cfg.depth.write_workers))
    prev_futs: list[Future] = []
    runtime_batch = [batch_size]  # halved on OOM so the run keeps going (P2-1)

    def collect(futs: list[Future]) -> None:
        """Wait for a batch's writes, absorb outcomes, persist the manifest."""
        n = 0
        for f in futs:
            path, ok, err = f.result()
            n += 1
            if ok:
                report.processed += 1
            else:
                report.failed += 1
                report.failures.append({"image": path, "task": "depth", "error": err})
        progress.advance(pid, n)
        if report.failed:
            progress.update(pid, description=f"depth · {report.failed} failed")
        manifest.save()

    def infer_results(items: list[tuple[Path, np.ndarray]]) -> list[dict]:
        imgs = [im for _p, im in items]
        try:
            return engine.infer_batch(imgs)
        except Exception:  # batch-level failure (OOM, size quirks): degrade
            runtime_batch[0] = max(1, runtime_batch[0] // 2)  # P2-1: remember the smaller batch
            console.print(
                f"[yellow]depth: batch inference failed, halving runtime batch to "
                f"{runtime_batch[0]} for the rest of this run[/]"
            )
            out: list[dict] = []
            for _p, im in items:
                try:
                    out.append(engine.infer(im))
                except Exception as exc:  # noqa: BLE001
                    out.append({"__error__": str(exc)})
            return out

    def submit_writes(items, results) -> list[Future]:
        futs: list[Future] = []
        for (path, _img), res in zip(items, results):
            if isinstance(res, dict) and "__error__" in res:
                manifest.mark(str(path), "depth", STATUS_FAILED, error=res["__error__"])
                report.failed += 1
                report.failures.append(
                    {"image": str(path), "task": "depth", "error": res["__error__"]}
                )
                progress.advance(pid, 1)
                continue
            futs.append(write_pool.submit(_depth_write_worker, write, manifest, path, res))
        return futs

    batch: list[tuple[Path, np.ndarray]] = []
    batch_portrait: bool | None = None

    def flush() -> None:
        nonlocal batch, prev_futs
        if not batch:
            return
        items, batch = batch, []
        results = infer_results(items)      # GPU busy
        if prev_futs:
            collect(prev_futs)              # previous batch's writes (likely done)
        prev_futs = submit_writes(items, results)

    while True:
        item = q.get()
        if item is sentinel:
            break
        path, img, err = item
        if err is not None:
            manifest.mark(str(path), "depth", STATUS_FAILED, error=str(err))
            report.failed += 1
            report.failures.append({"image": str(path), "task": "depth", "error": str(err)})
            progress.advance(pid, 1)
            continue
        portrait = _is_portrait(img)
        # orientation bucketing: same-orientation images batch together
        if batch and (len(batch) >= runtime_batch[0] or portrait != batch_portrait):
            flush()
        batch.append((path, img))
        batch_portrait = portrait
    flush()
    if prev_futs:
        collect(prev_futs)
    write_pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# detect phase: pipelined (prefetch decode | (image,prompt) pairs | async writes)
# ---------------------------------------------------------------------------
def make_detect_infer(cfg: RunConfig, engine: BaseEngine, prompts_by_class: list[tuple[str, str]]):
    """fn(image_path, rgb) -> detections. Labels are SAM3 prompts; the writer
    maps them to ontology class names (single mapping point)."""
    def infer(image_path: Path, rgb: np.ndarray) -> list[Detection]:
        return engine.infer(
            rgb,
            prompts=[p for _, p in prompts_by_class],
            conf_thres=cfg.detect.conf_thres,
            max_objects_per_prompt=cfg.detect.max_objects_per_prompt,
            nms_iou=cfg.detect.nms_iou,
        )

    return infer


def make_detect_writer(cfg: RunConfig, prompts_by_class: list[tuple[str, str]],
                       class_ids: dict[str, int]):
    """fn(image_path, rgb, dets) -> fragment for the COCO store.

    Writes ``semantic/<stem>.png`` (class-index), ``semantic_viz/<stem>.jpg``
    (color preview) and ``detect_viz/<stem>.jpg`` (overlay), and returns a
    COCO fragment (image entry + RLE annotations, standard fields only).

    dets arrive labeled with their SAM3 prompt; the prompt -> ontology class
    mapping (from prompts_by_class) is applied here — single mapping point.
    """
    prompt_to_class = {p: c for c, p in prompts_by_class}
    viz_palette = np.array([
        [0, 0, 0], [31, 119, 180], [255, 127, 14], [44, 160, 44],
        [214, 39, 40], [148, 103, 189], [140, 86, 75], [227, 119, 194],
    ], dtype=np.uint8)

    def write(image_path: Path, rgb: np.ndarray, dets: list[Detection]) -> dict:
        paths = out_paths(cfg.dataset.out_dir, image_path, cfg.dataset.image_dir)
        h, w = rgb.shape[:2]

        # dets arrive labeled with their SAM3 prompt -> map to ontology class
        for det in dets:
            det.label = prompt_to_class.get(det.label, det.label)

        canvas = np.zeros((h, w), dtype=np.uint8)
        anns: list[dict] = []
        for det in sorted(dets, key=lambda d: class_ids.get(d.label, 10_000)):
            if det.mask is None or not det.mask.any():
                continue
            cid = class_ids.get(det.label)
            if cid is None:
                continue
            canvas[det.mask] = cid
            ys, xs = np.nonzero(det.mask)
            anns.append({
                "category_id": cid,
                "segmentation": rle_encode(det.mask),
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                "area": int(det.mask.sum()),
                "iscrowd": 0,
                "score": round(float(det.score), 4),
            })

        from .io.depth_io import imwrite_safe

        imwrite_safe(paths["semantic_png"], canvas)
        imwrite_safe(paths["semantic_viz"], viz_palette[np.clip(canvas, 0, len(viz_palette) - 1)])
        if cfg.detect.viz:
            _write_image_jpg(paths["detect_viz"],
                             draw_detections(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dets,
                                             class_ids=class_ids))
        return {
            "image": {"file_name": _relative_name(image_path, cfg.dataset.image_dir),
                      "width": w, "height": h},
            "annotations": anns,
        }

    return write


def _detect_write_worker(write, manifest: Manifest, ontology_hash: str,
                         image_path: Path, rgb: np.ndarray, dets: list[Detection]):
    """Runs inside the write pool: semantic + viz, returns the COCO fragment."""
    try:
        fragment = write(image_path, rgb, dets)
        manifest.mark(str(image_path), "detect", STATUS_DONE,
                      n_objects=len(fragment["annotations"]), ontology_hash=ontology_hash)
        return str(image_path), True, None, fragment
    except Exception as exc:  # noqa: BLE001
        manifest.mark(str(image_path), "detect", STATUS_FAILED, error=str(exc))
        return str(image_path), False, str(exc), None


def _run_detect_phase(
    cfg: RunConfig,
    registry: ModelRegistry,
    images: list[Path],
    manifest: Manifest,
    report: RunReport,
    progress: Progress,
    pid,
    ontology_hash: str,
    coco_path: Path,
) -> None:
    """Pipelined detect phase: producer decode -> (image, prompt) pair
    inference -> thread pool writes semantic/viz; COCO fragments merge in the
    main thread and are throttled-dumped to ``detect/annotations.json``."""
    import queue

    coco = coco_load(coco_path)

    spec = registry.get(cfg.detect.model)
    engine = build_engine(spec.provider, spec.hf_id, _engine_params(spec))
    engine.load(device=cfg.detect.device, half=cfg.detect.half, source=cfg.download.source)
    from .core.ontology import load_ontology

    classes = load_ontology(cfg.detect.ontology)
    prompts_by_class = [(c.name, c.prompt) for c in classes]
    class_ids = {c.name: i + 1 for i, c in enumerate(classes)}  # 0 = background
    # categories 是导出的权威定义（ontology 顺序，id 从 1 开始）
    coco["categories"] = [
        {"id": i + 1, "name": c.name, "supercategory": ""}
        for i, c in enumerate(classes)
    ]
    infer = make_detect_infer(cfg, engine, prompts_by_class)
    write = make_detect_writer(cfg, prompts_by_class, class_ids)
    prompt_batch = max(1, int(getattr(engine, "prompt_batch", 4)))
    n_prompts = max(1, len(prompts_by_class))

    q: queue.Queue = queue.Queue(maxsize=max(2, cfg.detect.prefetch))
    sentinel = object()

    def produce() -> None:
        for p in images:
            try:
                q.put((p, _read_image_rgb(p), None))
            except Exception as exc:  # noqa: BLE001
                q.put((p, None, exc))
        q.put(sentinel)

    threading.Thread(target=produce, daemon=True).start()

    write_pool = ThreadPoolExecutor(max_workers=max(1, cfg.detect.write_workers))
    futs: list = []

    def collect() -> None:
        n = 0
        for f in futs:
            path, ok, err, fragment = f.result()
            n += 1
            if ok:
                report.processed += 1
                if fragment:
                    image_id = coco_upsert_image(coco, **fragment["image"])
                    for ann in fragment["annotations"]:
                        ann["image_id"] = image_id
                        ann["id"] = coco_next_annotation_id(coco)
                        coco["annotations"].append(ann)
            else:
                report.failed += 1
                report.failures.append({"image": path, "task": "detect", "error": err})
        futs.clear()
        progress.advance(pid, n)
        manifest.maybe_save(5.0)
        coco_save(coco, coco_path)

    pairs: list[tuple[Path, np.ndarray, str]] = []  # (path, rgb, prompt)
    states: dict[int, dict] = {}

    def flush_chunk() -> None:
        nonlocal pairs
        if not pairs:
            return
        chunk, pairs = pairs[:prompt_batch], pairs[prompt_batch:]
        images = [im for _p, im, _t in chunk]
        texts = [t for _p, _im, t in chunk]
        try:
            results = engine.infer_pairs(
                images, texts,
                conf_thres=cfg.detect.conf_thres,
                max_objects_per_prompt=cfg.detect.max_objects_per_prompt,
                nms_iou=cfg.detect.nms_iou,
            )
        except Exception:  # batch-level failure (OOM): degrade to singles
            results = []
            for _p, im, t in chunk:
                try:
                    results.append(engine.infer(im, prompts=[t],
                                                conf_thres=cfg.detect.conf_thres,
                                                max_objects_per_prompt=cfg.detect.max_objects_per_prompt,
                                                nms_iou=cfg.detect.nms_iou))
                except Exception as exc:  # noqa: BLE001
                    results.append(exc)
        for (path, _rgb, _prompt), res in zip(chunk, results):
            state = states[id(path)]
            if isinstance(res, Exception):
                state["errors"].append(str(res))
            else:
                state["dets"].extend(res)
            state["remaining"] -= 1
            if state["remaining"] == 0:
                if state["errors"]:
                    err = "; ".join(state["errors"])
                    manifest.mark(str(path), "detect", STATUS_FAILED, error=err)
                    report.failed += 1
                    report.failures.append({"image": str(path), "task": "detect", "error": err})
                else:
                    futs.append(write_pool.submit(
                        _detect_write_worker, write, manifest, ontology_hash,
                        state["path"], state["rgb"], state["dets"],
                    ))

    while True:
        item = q.get()
        if item is sentinel:
            break
        path, rgb, err = item
        if err is not None:
            manifest.mark(str(path), "detect", STATUS_FAILED, error=str(err))
            report.failed += 1
            report.failures.append({"image": str(path), "task": "detect", "error": str(err)})
            progress.advance(pid, 1)
            continue
        idx = id(path)
        states[idx] = {"path": path, "rgb": rgb, "dets": [],
                       "remaining": n_prompts, "errors": []}
        for _c, _p in prompts_by_class:
            pairs.append((path, rgb, _p))
        while len(pairs) >= prompt_batch:
            before = len(pairs)
            flush_chunk()
            if len(pairs) == before:  # safety: never spin without progress
                pairs.clear()
    # stream end: flush remaining pairs and collect writes
    while pairs:
        before = len(pairs)
        flush_chunk()
        if len(pairs) == before:
            break
    if futs:
        collect()
    write_pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def _disk_free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(str(p)).free


def ontology_mismatch_count(cfg: RunConfig, manifest: Manifest) -> int:
    """Detect records whose labels predate an ontology.yaml change."""
    onto = Path(cfg.detect.ontology)
    if not onto.exists():
        return 0
    current = _ontology_hash(cfg)
    return sum(
        1 for rec in manifest.records.values()
        if rec.get("tasks", {}).get("detect", {}).get("status") == STATUS_DONE
        and rec["tasks"]["detect"].get("ontology_hash") not in (None, current)
    )


def _ontology_hash(cfg: RunConfig) -> str:
    onto = Path(cfg.detect.ontology)
    if not onto.exists():
        return "n/a"
    return hashlib.sha1(onto.read_bytes()).hexdigest()[:12]


def run_pipeline(
    cfg: RunConfig,
    tasks: list[str] | None = None,
    resume: bool = True,
    limit: int | None = None,
    redo_tasks: list[str] | None = None,
    force: bool = False,
) -> RunReport:
    tasks = tasks or cfg.tasks
    registry = ModelRegistry.load()
    images = scan_images(cfg.dataset.image_dir, cfg.dataset.patterns)
    if limit:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"No images found under {cfg.dataset.image_dir}")

    # P2-2: cheap disk-space precheck (estimate ~2.2x input volume for outputs)
    if not force:
        est = sum(p.stat().st_size for p in images) * 2.2
        free = _disk_free_bytes(cfg.dataset.out_dir)
        if est > free:
            raise RuntimeError(
                f"Estimated output ~{est / 1e9:.1f} GB exceeds {free / 1e9:.1f} GB free "
                f"at {cfg.dataset.out_dir}. Free space or rerun with --force."
            )

    manifest = Manifest(out_paths(cfg.dataset.out_dir, images[0])["manifest"], lock=True)
    report = RunReport(total=len(images))

    # P1-7: --redo-detect clears detect statuses so resume re-runs them
    redo = set(redo_tasks or [])
    if redo:
        for rec in manifest.records.values():
            for task in redo:
                rec.get("tasks", {}).pop(task, None)
        manifest.save()

    # P1-7: warn when done detect labels predate an ontology.yaml change
    if resume and "detect" in tasks:
        stale = ontology_mismatch_count(cfg, manifest)
        if stale:
            console.print(
                f"[yellow]WARNING: {stale} detect labels were produced with a different "
                f"ontology.yaml. Rerun with --redo-detect to refresh them.[/]"
            )

    console.print(
        f"[bold]AssistLabel[/] | images: [cyan]{len(images)}[/] | tasks: [cyan]{','.join(tasks)}[/] "
        f"| depth batch: [cyan]{cfg.depth.batch_size}[/] | source: [cyan]{cfg.download.source}[/] "
        f"| resume: {resume}"
    )
    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    try:
        for task in tasks:
            pending = [im for im in images if not (resume and manifest.is_done(str(im), task))]
            report.skipped += len(images) - len(pending)
            if not pending:
                continue
            with Progress(*columns, console=console) as progress:
                pid = progress.add_task(task, total=len(pending))
                if task == "depth":
                    _run_depth_phase(cfg, registry, pending, manifest, report, progress, pid)
                elif task == "detect":
                    _run_detect_phase(cfg, registry, pending, manifest, report, progress, pid,
                                      ontology_hash=_ontology_hash(cfg),
                                      coco_path=Path(cfg.dataset.out_dir) / "detect" / "annotations.json")
                else:
                    raise EngineError(f"Unknown task: {task}")
                if report.failed:
                    progress.update(pid, description=f"{task} · {report.failed} failed")
            # unload phase engines so the next phase can take the GPU
            for provider in ("depth_anything", "depth_anything3", "sam3"):
                engine = _ENGINE_CACHE.get(provider)
                if engine is not None:
                    engine.unload()
    finally:
        manifest.save()  # never lose resume state, even on Ctrl+C
        manifest.release()

    return report
