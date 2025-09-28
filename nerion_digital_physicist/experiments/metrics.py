"""Metrics aggregation utilities for Nerion Phase 3."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from ..infrastructure.memory import ReplayStore
from ..infrastructure.registry import CATALOG_FILENAME
from ..infrastructure.telemetry import TELEMETRY_FILENAME


@dataclass
class MetricSummary:
    total_tasks: int
    templates: Dict[str, int]
    replay_status_counts: Dict[str, int]
    average_surprise: Optional[float]
    average_duration: Optional[float]
    generation_runs: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _collect_manifest_stats(root: Path) -> (int, Dict[str, int]):
    catalog_entries = _load_jsonl(root / CATALOG_FILENAME)
    templates: Dict[str, int] = {}
    for entry in catalog_entries:
        template_id = entry.get("template_id", "unknown")
        templates[template_id] = templates.get(template_id, 0) + 1
    return len(catalog_entries), templates


def _collect_replay_stats(root: Path) -> (Dict[str, int], List[float]):
    store = ReplayStore(root)
    status_counts: Dict[str, int] = {}
    surprises: List[float] = []
    for exp in store.load():
        status_counts[exp.status] = status_counts.get(exp.status, 0) + 1
        if exp.surprise is not None:
            surprises.append(float(exp.surprise))
    return status_counts, surprises


def _collect_telemetry_durations(root: Path) -> (int, List[float]):
    events = _load_jsonl(root / TELEMETRY_FILENAME)
    durations: List[float] = []
    run_events = 0
    for entry in events:
        if entry.get("event_type") == "generation_run_complete":
            payload = entry.get("payload", {})
            run_events += 1
            if "duration_seconds" in payload:
                durations.append(float(payload["duration_seconds"]))
    return run_events, durations


def summarize_metrics(root: Path) -> MetricSummary:
    total_tasks, template_counts = _collect_manifest_stats(root)
    replay_status_counts, surprises = _collect_replay_stats(root)
    run_events, durations = _collect_telemetry_durations(root)

    avg_surprise = mean(surprises) if surprises else None
    avg_duration = mean(durations) if durations else None

    return MetricSummary(
        total_tasks=total_tasks,
        templates=template_counts,
        replay_status_counts=replay_status_counts,
        average_surprise=avg_surprise,
        average_duration=avg_duration,
        generation_runs=run_events,
    )
