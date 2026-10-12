"""Manifest 并发安全：write 线程 mark() 与主线程 save() 同时进行。"""

import json
import threading

from assistlabel.core.pipeline import Manifest


def test_concurrent_mark_and_save(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    errors: list[Exception] = []

    def marker(base: int) -> None:
        try:
            for i in range(300):
                m.mark(f"img_{base}_{i}.png", "detect", "done", n_objects=i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def saver() -> None:
        try:
            for _ in range(200):
                m.save()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=marker, args=(b,)) for b in range(4)]
    threads.append(threading.Thread(target=saver))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent mark/save raised: {errors[0]!r}"
    assert len(m.records) == 1200
    assert not errors
    # reload round-trips everything that was saved
    m2 = Manifest(tmp_path / "manifest.jsonl")
    assert len(m2.records) >= 1
    rec = m2.records["img_0_299.png"]
    assert json.loads(json.dumps(rec))["tasks"]["detect"]["status"] == "done"
