"""Append-only memory journal (internal use).

This module provides a tiny JSONL-backed event store for memory mutations.
It deliberately favours durability and simplicity over complex indexing – the
expected write volume is low and bounded by memory operations.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional

from ops.security.fs_guard import ensure_in_repo_auto

try:  # Runtime optional: telemetry bus can fail without blocking journaling
    from ops.telemetry import TelemetryEvent, ensure_default_sinks, publish as _publish_event

    ensure_default_sinks()
except Exception:  # pragma: no cover - telemetry is optional
    TelemetryEvent = None  # type: ignore

    def ensure_default_sinks(*_a, **_kw):  # type: ignore
        return None

    def _publish_event(*_a, **_kw):  # type: ignore
        return None

TELEMETRY_SOURCE = "core.memory.journal_store"

_JOURNAL_PATH = ensure_in_repo_auto(Path("out/memory/memory_journal.jsonl"))
_JOURNAL_LOCK = RLock()


def _ensure_path() -> Path:
    path = Path(_JOURNAL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_lines(limit: Optional[int] = None) -> Iterable[str]:
    path = _ensure_path()
    if not path.exists():
        return []
    if limit is None:
        with path.open("r", encoding="utf-8") as handle:
            return list(handle)
    window: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            window.append(line)
    return list(window)


def _load_events(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for raw in _read_lines(limit):
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events


def log_event(kind: str, rationale: str = "", **fields: Any) -> Dict[str, Any]:
    """Append an event to the journal and return it."""
    event: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kind": kind,
        "rationale": rationale,
    }
    if fields:
        event.update(fields)
    payload = json.dumps(event, ensure_ascii=False)
    with _JOURNAL_LOCK:
        path = _ensure_path()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")

    _publish_to_telemetry(event)
    return event


def last() -> Optional[Dict[str, Any]]:
    events = _load_events(limit=1)
    return events[-1] if events else None


def timeline(limit: int = 50, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    items = _load_events()
    if kind is not None:
        items = [e for e in items if e.get("kind") == kind]
    return items[-max(0, limit):]


def export(path: Any) -> None:
    out_path = ensure_in_repo_auto(Path(path))
    events = _load_events()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as handle:
        json.dump(events, handle, ensure_ascii=False, indent=2)


def query(
    kind: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    contains: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    events = _load_events()
    results: List[Dict[str, Any]] = []
    for event in events:
        if kind is not None and event.get("kind") != kind:
            continue
        timestamp = str(event.get("timestamp", ""))
        if since is not None and timestamp < since:
            continue
        if until is not None and timestamp > until:
            continue
        if contains is not None:
            blob = json.dumps(event, ensure_ascii=False).lower()
            if contains.lower() not in blob:
                continue
        results.append(event)
    return results[-max(0, limit):]


def migrate_legacy_jsonl(legacy_path: Optional[str] = None) -> int:
    """Best-effort migration from the previous JSON-array format."""
    guess = Path("memory_journal.json") if legacy_path is None else Path(legacy_path)
    guess = Path(ensure_in_repo_auto(guess))
    if not guess.exists():
        return 0
    migrated = 0
    try:
        with guess.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    events = data.get("events")
    if not isinstance(events, list):
        return 0
    for entry in events:
        if isinstance(entry, dict):
            log_event(entry.get("kind", "legacy"), entry.get("rationale", ""), **{k: v for k, v in entry.items() if k not in {"kind", "rationale"}})
            migrated += 1
    return migrated


try:
    migrate_legacy_jsonl()
except Exception:
    pass


def _publish_to_telemetry(event: Dict[str, Any]) -> None:
    if TelemetryEvent is None:
        return

    kind = event.get("kind", "other")
    rationale = str(event.get("rationale", ""))
    subject = _infer_subject(event)
    metadata, payload = _partition_fields(event, exclude_keys={"payload", "metadata"})
    metadata.setdefault("rationale", rationale)

    telemetry_event = TelemetryEvent(
        kind=kind if isinstance(kind, str) else str(kind),
        source=TELEMETRY_SOURCE,
        subject=subject,
        metadata=metadata,
        payload=payload or None,
        tags=["journal"],
    )

    redact_keys = []
    if not _bool_env("NERION_V2_LOG_PROMPTS", False):
        redact_keys.extend(["prompt", "completion", "diff"])

    try:
        _publish_event(telemetry_event, redact_keys=redact_keys or None, extra_tags=None, auto_flush=True)
    except Exception:  # pragma: no cover - bus failures must not break journaling
        pass


def _infer_subject(event: Dict[str, Any]) -> Optional[str]:
    for key in ("subject", "plan_path", "snapshot_id", "file", "path", "task", "interaction_id"):
        if key in event and event[key] is not None:
            try:
                return str(event[key])
            except Exception:
                return None
    return None


def _partition_fields(event: Dict[str, Any], exclude_keys: set[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    payload: Dict[str, Any] = {}
    for key, value in event.items():
        if key in {"id", "timestamp", "kind"} | exclude_keys:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        else:
            try:
                json.dumps(value, ensure_ascii=False)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
    return metadata, payload


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}
