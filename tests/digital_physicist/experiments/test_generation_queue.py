from pathlib import Path

from nerion_digital_physicist.generation.queue_manager import GenerationQueue, QueueItem


def test_enqueue_dequeue_roundtrip(tmp_path: Path) -> None:
    queue = GenerationQueue(tmp_path)
    item = QueueItem(request_id="req1", count=3)
    queue.enqueue(item)

    assert len(queue) == 1
    retrieved = queue.peek()
    assert retrieved.request_id == "req1"

    dequeued = queue.dequeue()
    assert dequeued.request_id == "req1"
    assert len(queue) == 0


def test_queue_persistence(tmp_path: Path) -> None:
    queue = GenerationQueue(tmp_path)
    queue.enqueue(QueueItem(request_id="req2", count=2))

    queue2 = GenerationQueue(tmp_path)
    assert len(queue2) == 1
    assert queue2.peek().request_id == "req2"
