"""Telemetry package entrypoint."""

from __future__ import annotations

import os
from typing import Optional

from .bus import TelemetryBus, TelemetrySink, get_bus, publish, publish_dict
from .schema import EventKind, TelemetryEvent
from .store import TelemetryStore
from .snapshots import write_snapshot
from .reflection import ReflectionConfig
from .reflection_job import run_reflection
from .operator import get_latest_reflection, load_operator_snapshot, summarize_snapshot
from .experiment_journal import (
    ExperimentRecord,
    create_experiment,
    list_experiments,
    update_experiment,
    get_experiment,
)
from .knowledge_graph import (
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    build_knowledge_graph,
    knowledge_hotspots,
    load_knowledge_graph,
    summarize_knowledge_graph,
    write_knowledge_graph,
)

_DEFAULT_SINK_REGISTERED = False
_DEFAULT_SINKS: list[TelemetrySink] = []


def _bool_env(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def ensure_default_sinks(path: Optional[str] = None) -> None:
    """Attach the built-in JSONL sink unless disabled via env."""

    global _DEFAULT_SINK_REGISTERED
    global _DEFAULT_SINKS

    if _DEFAULT_SINK_REGISTERED:
        return

    from .sinks import JsonlSink, SQLiteSink

    sinks: list[TelemetrySink] = []
    if _bool_env("NERION_V2_TELEMETRY_JSONL", True):
        sink_path = path or os.getenv("NERION_V2_TELEMETRY_JSONL_PATH") or None
        sinks.append(JsonlSink(sink_path))
    if _bool_env("NERION_V2_TELEMETRY_SQLITE", True):
        sqlite_path = os.getenv("NERION_V2_TELEMETRY_SQLITE_PATH") or None
        sinks.append(SQLiteSink(sqlite_path))

    for sink in sinks:
        get_bus().subscribe(sink)

    _DEFAULT_SINKS = sinks
    _DEFAULT_SINK_REGISTERED = True


__all__ = [
    "EventKind",
    "TelemetryEvent",
    "TelemetryBus",
    "TelemetrySink",
    "TelemetryStore",
    "write_snapshot",
    "ReflectionConfig",
    "run_reflection",
    "get_latest_reflection",
    "load_operator_snapshot",
    "summarize_snapshot",
    "ExperimentRecord",
    "create_experiment",
    "update_experiment",
    "list_experiments",
    "get_experiment",
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "build_knowledge_graph",
    "load_knowledge_graph",
    "knowledge_hotspots",
    "write_knowledge_graph",
    "summarize_knowledge_graph",
    "ensure_default_sinks",
    "get_bus",
    "publish",
    "publish_dict",
]
