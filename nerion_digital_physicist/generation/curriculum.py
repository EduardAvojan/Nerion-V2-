"""Curriculum policies derived from telemetry logs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from ..infrastructure.telemetry import TELEMETRY_FILENAME


@dataclass
class TemplateStats:
    template_id: str
    count: int = 0
    total_duration: float = 0.0

    @property
    def avg_duration(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_duration / self.count


def iter_telemetry_events(root: Path) -> Iterable[dict]:
    file_path = root / TELEMETRY_FILENAME
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_template_stats(root: Path) -> Dict[str, TemplateStats]:
    stats: Dict[str, TemplateStats] = {}
    for event in iter_telemetry_events(root):
        if event.get("event_type") != "task_generated":
            continue
        payload = event.get("payload", {})
        template_id = payload.get("template_id")
        if not template_id:
            continue
        stats.setdefault(template_id, TemplateStats(template_id)).count += 1
        stats[template_id].total_duration += float(payload.get("duration_seconds", 0.0))
    return stats


def compute_curriculum_weights(
    root: Path,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
) -> Dict[str, float]:
    stats = collect_template_stats(root)
    if not stats:
        return {}

    total_count = sum(s.count for s in stats.values())
    total_duration = sum(s.total_duration for s in stats.values())
    avg_duration_overall = total_duration / total_count if total_count else 0.0

    weights: Dict[str, float] = {}
    for template_id, detail in stats.items():
        rarity = 1.0
        if total_count:
            frequency = detail.count / total_count
            rarity = 1.0 - frequency  # favor less frequent templates
        duration_multiplier = 1.0
        if avg_duration_overall > 0:
            duration_multiplier = detail.avg_duration / avg_duration_overall
        raw_weight = rarity + 0.5 * duration_multiplier
        clamped = max(min_weight, min(max_weight, raw_weight))
        weights[template_id] = clamped
    return weights
