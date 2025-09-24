"""Outcome logging utilities for replay integration."""
from __future__ import annotations

from typing import Any, Dict

from .memory import ReplayStore
from .telemetry import TelemetryLogger


def log_outcome(
    replay: ReplayStore,
    telemetry: TelemetryLogger | None,
    experience_id: str | None = None,
    task_id: str | None = None,
    status: str = "pending",
    surprise: float | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> None:
    if not experience_id:
        if not task_id:
            raise ValueError("Either experience_id or task_id must be provided")
        exp = replay.find_by_task(task_id)
        if not exp:
            raise KeyError(f"No experience found for task_id {task_id}")
        experience_id = exp.experience_id

    updated = replay.update(
        experience_id,
        status=status,
        surprise=surprise,
        priority=replay._default_priority(surprise),
        metadata=extra_metadata or {},
    )

    if telemetry:
        telemetry.log(
            "experience_updated",
            {
                "experience_id": experience_id,
                "task_id": updated.task_id,
                "template_id": updated.template_id,
                "status": updated.status,
                "surprise": updated.surprise,
                "priority": updated.priority,
            },
        )
