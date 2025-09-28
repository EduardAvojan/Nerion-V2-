"""Telemetry reflection pipeline (design scaffolding).

The reflection job summarises recent telemetry into structured artefacts that
other subsystems (planner, HOLO dashboards) can consume. It operates entirely
offline, using the local telemetry SQLite store and writing JSON summaries to
`out/telemetry/reflections/`.

Inputs
------
* Rolling window of telemetry events (default: last 24h, configurable).
* Namespace filters (e.g., include only provider/completion events when
  computing latency drift).
* Optional embeddings: a VectorStore namespace can house textual
  representations of prompts/responses for semantic clustering.

Outputs
-------
* `summary` block: top-level counts, latency deltas, error spikes, provider
  cost estimates.
* `clusters`: groups of similar prompts/errors for follow-up (each references
  example event ids, tags, timestamps).
* `anomalies`: notable deviations (e.g., spike in completion failures) with
  simple heuristics.
* `metadata`: run configuration (window_start/end, event_count, git commit,
  planner defaults).

Storage Contract
----------------
* JSON files named `reflection_<ts>.json` under `out/telemetry/reflections/`.
* optional vector embeddings stored in VectorStore namespace `reflections` for
  semantic recall of past findings.

This module defines the minimal data model and configuration helpers. The
execution logic lives alongside the job implementation to keep responsibilities
clear: this file should remain importable without triggering heavy work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_WINDOW_HOURS = 24
REFLECTION_DIR = Path("out/telemetry/reflections")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ReflectionConfig:
    """Runtime configuration for a reflection run."""

    window_hours: int = DEFAULT_WINDOW_HOURS
    limit_events: int = 5000
    include_kinds: Optional[Iterable[str]] = None
    vector_namespace: Optional[str] = "reflections"
    include_memory: bool = True

    def window_bounds(self) -> tuple[str, str]:
        end = utc_now()
        start = end - timedelta(hours=max(1, self.window_hours))
        return (start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z"))


@dataclass
class ClusterSummary:
    tag: str
    event_ids: List[str] = field(default_factory=list)
    example: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    kind: str
    detail: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionSummary:
    timestamp: str
    window_start: str
    window_end: str
    event_count: int
    summary: Dict[str, Any] = field(default_factory=dict)
    clusters: List[ClusterSummary] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def reflection_output_path(ts: datetime | None = None) -> Path:
    ts = ts or utc_now()
    safe = ts.isoformat().replace("+00:00", "Z").replace(":", "").replace("-", "").replace(".", "")
    directory = Path(REFLECTION_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"reflection_{safe}.json"


__all__ = [
    "ReflectionConfig",
    "ClusterSummary",
    "Anomaly",
    "ReflectionSummary",
    "reflection_output_path",
    "REFLECTION_DIR",
]
