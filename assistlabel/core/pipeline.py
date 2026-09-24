"""Manifest-driven batch pipeline with crash-safe resume.

``manifest.jsonl`` holds one JSON line per image: path, content hash, per-task
status/timings and result counts. The file is rewritten atomically (tmp +
rename) after every image, so an interrupted run can be resumed with
``--resume`` and completed images are skipped.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"


class ManifestLockedError(RuntimeError):
    """Raised when another process holds the manifest lock."""


class Manifest:
    def __init__(self, path: str | Path, lock: bool = False):
        self.path = Path(path)
        self.records: dict[str, dict] = {}  # keyed by image path string
        self._last_save = 0.0
        self._lock_fh = None
        if lock:
            self._acquire()
        self.load()

    def _acquire(self) -> None:
        """Exclusive non-blocking lock on ``<manifest>.lock``.

        Prevents two concurrent ``run`` processes from silently overwriting
        each other's progress. The lock file is created next to the manifest
        and is harmless to delete once no instance is running.
        """
        lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+")
        try:
            fh.write("0")
            fh.flush()
            fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fh = fh
        except OSError as exc:
            fh.close()
            raise ManifestLockedError(
                f"Manifest is locked by another AssistLabel process ({lock_path}). "
                "Stop it or delete the .lock file if it was left behind by a crash."
            ) from exc

    def release(self) -> None:
        if self._lock_fh is None:
            return
        try:
            self._lock_fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_fh.close()
            self._lock_fh = None

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        self.records = {}
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line from a crash
                if isinstance(rec, dict) and "image" in rec:
                    self.records[rec["image"]] = rec

    def save(self) -> None:
        import time as _time

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".manifest_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for rec in self.records.values():
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)  # atomic on POSIX and Windows
            self._last_save = _time.monotonic()
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def maybe_save(self, interval: float = 5.0) -> None:
        """Save at most once per ``interval`` seconds.

        The save is a full rewrite, O(number of images); on million-image
        sets saving after every image would dominate runtime. Losing the last
        few seconds of status after a crash is harmless — affected images are
        simply reprocessed on resume.
        """
        import time as _time

        if _time.monotonic() - self._last_save >= interval:
            self.save()

    # -- record access -----------------------------------------------------
    def get(self, image_path: str) -> dict:
        return self.records.setdefault(
            image_path, {"image": image_path, "tasks": {}, "total_objects": 0}
        )

    def mark(self, image_path: str, task: str, status: str, **extra) -> dict:
        rec = self.get(image_path)
        entry = {"status": status, **extra}
        rec["tasks"][task] = entry
        return entry

    def task_status(self, image_path: str, task: str) -> str | None:
        entry = self.records.get(image_path, {}).get("tasks", {}).get(task)
        return entry.get("status") if entry else None

    def is_done(self, image_path: str, task: str) -> bool:
        return self.task_status(image_path, task) == STATUS_DONE

    def summary(self) -> dict:
        per_task: dict[str, dict[str, int]] = {}
        for rec in self.records.values():
            for task, entry in rec.get("tasks", {}).items():
                bucket = per_task.setdefault(task, {STATUS_DONE: 0, STATUS_FAILED: 0})
                if entry.get("status") in bucket:
                    bucket[entry["status"]] += 1
        return {"images": len(self.records), "tasks": per_task}


@dataclass
class RunReport:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    seconds: float = 0.0
    failures: list[dict] = field(default_factory=list)

    def merge(self, other: "RunReport") -> None:
        self.total += other.total
        self.processed += other.processed
        self.skipped += other.skipped
        self.failed += other.failed
        self.seconds += other.seconds
        self.failures.extend(other.failures)


def run_image_tasks(
    image_path: Path,
    tasks: list[str],
    *,
    depth_runner=None,
    detect_runner=None,
    fuse_runner=None,
    manifest: Manifest | None = None,
    resume: bool = True,
) -> RunReport:
    """Run the requested tasks for one image.

    ``*_runner`` are callables ``fn(image_path) -> dict`` that do the actual
    work for each task; they are injected by the pipeline orchestrator so this
    function stays engine-agnostic and unit-testable with mock runners.
    """
    report = RunReport(total=1)
    runners = {"depth": depth_runner, "detect": detect_runner, "fuse": fuse_runner}
    key = str(image_path)
    start = time.perf_counter()

    for task in tasks:
        runner = runners[task]
        if runner is None:
            continue
        if resume and manifest is not None and manifest.is_done(key, task):
            report.skipped += 1
            continue
        t0 = time.perf_counter()
        try:
            result = runner(image_path) or {}
            elapsed = round(time.perf_counter() - t0, 3)
            if manifest is not None:
                result.pop("seconds", None)  # reserved: measured here
                manifest.mark(key, task, STATUS_DONE, seconds=elapsed, **result)
        except Exception as exc:  # noqa: BLE001 - batch must not die on one image
            elapsed = round(time.perf_counter() - t0, 3)
            report.failed += 1
            report.failures.append({"image": key, "task": task, "error": str(exc)})
            if manifest is not None:
                manifest.mark(key, task, STATUS_FAILED, seconds=elapsed, error=str(exc))
        else:
            report.processed += 1
            n = result.get("n_objects")
            if task == "detect" and isinstance(n, int) and manifest is not None:
                manifest.get(key)["total_objects"] = n

    report.seconds = time.perf_counter() - start
    return report
