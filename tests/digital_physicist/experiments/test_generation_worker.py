from pathlib import Path

import threading

from nerion_digital_physicist.generation.worker import enqueue_request, process_next
from nerion_digital_physicist.infrastructure.registry import CATALOG_FILENAME
from nerion_digital_physicist.infrastructure.telemetry import TELEMETRY_FILENAME


def test_worker_generates_tasks(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_root = tmp_path / "output"
    enqueue_request(queue_root, count=2)

    processed = process_next(queue_root, output_root, seed=42, worker_id="worker-test")
    assert processed

    catalog_path = output_root / CATALOG_FILENAME
    lines = catalog_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    telemetry_lines = (output_root / TELEMETRY_FILENAME).read_text(encoding="utf-8").strip().splitlines()
    assert any("worker_started" in line for line in telemetry_lines)
    assert any("worker_finished" in line for line in telemetry_lines)


def test_worker_handles_empty_queue(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_root = tmp_path / "output"

    processed = process_next(queue_root, output_root)
    assert processed is False


def test_concurrent_workers_process_each_item_once(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    output_root = tmp_path / "output"
    enqueue_request(queue_root, count=1)

    results = []

    def worker(name: str):
        processed = process_next(queue_root, output_root, worker_id=name)
        results.append(processed)

    threads = [threading.Thread(target=worker, args=("w1",)), threading.Thread(target=worker, args=("w2",))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1
