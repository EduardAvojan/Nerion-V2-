"""DEPRECATED: Worker harness for processing generation queue items.

This template-based generation system has been deprecated in favor of LLM-based generators.
See DEPRECATED_TEMPLATE_SYSTEM.md for details.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from .queue_manager import GenerationQueue, QueueItem
from .service import load_template_specs
from .sampler import TemplateSampler
from .builder import TaskBuilder
from ..infrastructure.registry import ManifestRegistry
from ..infrastructure.telemetry import TelemetryLogger
from ..infrastructure.memory import ReplayStore


def process_next(queue_root: Path, output_root: Path, seed: int = 0, worker_id: str | None = None) -> bool:
    queue = GenerationQueue(queue_root)
    item = queue.dequeue()
    if not item:
        return False

    registry = ManifestRegistry(output_root)
    telemetry = TelemetryLogger(output_root)
    replay = ReplayStore(output_root)

    if worker_id is None:
        worker_id = uuid.uuid4().hex

    telemetry.log(
        "worker_started",
        {
            "worker_id": worker_id,
            "request_id": item.request_id,
            "count": item.count,
        },
    )

    specs = load_template_specs(item.template_weights)
    sampler = TemplateSampler(specs, seed=seed)
    builder = TaskBuilder(output_root, registry, telemetry=telemetry, replay=replay)

    processed = 0
    for spec in sampler.sequence(item.count):
        builder.build_task(spec.template_id, seed=seed)
        processed += 1

    telemetry.log(
        "worker_finished",
        {
            "worker_id": worker_id,
            "request_id": item.request_id,
            "processed": processed,
        },
    )
    return True


def enqueue_request(queue_root: Path, count: int, template_weights=None) -> QueueItem:
    queue = GenerationQueue(queue_root)
    request = QueueItem(request_id=uuid.uuid4().hex, count=count, template_weights=template_weights)
    queue.enqueue(request)
    return request
