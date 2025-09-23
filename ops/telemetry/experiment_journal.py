"""Experiment journal utilities for Phase 2 reflection & analysis.

Provides a lightweight JSON-backed registry so operators (or automation)
can record hypotheses, experiment plans, and outcomes while linking to the
latest telemetry reflections. The file lives under `out/telemetry` to keep
all Phase 2 artefacts together and remains local-only by default.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional

from ops.security.fs_guard import ensure_in_repo_auto

try:  # optional telemetry event for journaling
    from .schema import TelemetryEvent, EventKind
    from . import publish
except Exception:  # pragma: no cover - telemetry bus failure must not break journal
    TelemetryEvent = None  # type: ignore
    EventKind = None  # type: ignore

    def publish(*_args, **_kwargs):  # type: ignore
        return None

_EXPERIMENT_PATH = ensure_in_repo_auto(Path("out/telemetry/experiments.json"))
_LOCK = RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class ExperimentRecord:
    id: str
    title: str
    hypothesis: str
    status: str = "planned"
    reflection_path: Optional[str] = None
    reflection_timestamp: Optional[str] = None
    arms: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # ensure lists are json-serialisable even if dataclass enforces tuples etc
        data["arms"] = list(self.arms)
        data["tags"] = list(self.tags)
        data["metrics"] = dict(self.metrics)
        return data


def _ensure_file() -> Path:
    path = Path(_EXPERIMENT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")
    return path


def _read_all() -> List[ExperimentRecord]:
    path = _ensure_file()
    with path.open("r", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError:
            raw = []
    records: List[ExperimentRecord] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            records.append(
                ExperimentRecord(
                    id=str(entry.get("id") or uuid.uuid4()),
                    title=str(entry.get("title") or "Untitled"),
                    hypothesis=str(entry.get("hypothesis") or ""),
                    status=str(entry.get("status") or "planned"),
                    reflection_path=entry.get("reflection_path"),
                    reflection_timestamp=entry.get("reflection_timestamp"),
                    arms=list(entry.get("arms") or []),
                    metrics=dict(entry.get("metrics") or {}),
                    outcome=entry.get("outcome"),
                    notes=entry.get("notes"),
                    tags=list(entry.get("tags") or []),
                    created_at=str(entry.get("created_at") or _now_iso()),
                    updated_at=str(entry.get("updated_at") or _now_iso()),
                )
            )
    return records


def _write_all(records: Iterable[ExperimentRecord]) -> None:
    path = _ensure_file()
    payload = [record.to_dict() for record in records]
    with _LOCK:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _emit_telemetry(event: ExperimentRecord, action: str) -> None:
    if TelemetryEvent is None or EventKind is None:
        return
    try:
        payload = event.to_dict()
        telemetry = TelemetryEvent(
            kind=EventKind.METRIC,
            source="ops.telemetry.experiment_journal",
            subject=action,
            metadata={
                "id": event.id,
                "status": event.status,
                "title": event.title,
            },
            payload=payload,
            tags=["experiment", action],
        )
        publish(telemetry)
    except Exception:  # pragma: no cover - journaling must not crash
        return


def list_experiments(limit: Optional[int] = None) -> List[ExperimentRecord]:
    records = sorted(_read_all(), key=lambda item: item.updated_at, reverse=True)
    if limit is not None and limit > 0:
        return records[:limit]
    return records


def create_experiment(
    *,
    title: str,
    hypothesis: str,
    reflection_path: Optional[str] = None,
    reflection_timestamp: Optional[str] = None,
    arms: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    notes: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    status: str = "planned",
) -> ExperimentRecord:
    record = ExperimentRecord(
        id=uuid.uuid4().hex,
        title=title.strip() or "Untitled",
        hypothesis=hypothesis.strip(),
        status=status.strip() or "planned",
        reflection_path=reflection_path,
        reflection_timestamp=reflection_timestamp,
        arms=list(arms or []),
        tags=[tag.strip() for tag in (tags or []) if tag],
        notes=notes.strip() if isinstance(notes, str) else notes,
        metrics=dict(metrics or {}),
    )

    with _LOCK:
        records = _read_all()
        records.append(record)
        _write_all(records)

    _emit_telemetry(record, "created")
    return record


def update_experiment(experiment_id: str, **fields: Any) -> Optional[ExperimentRecord]:
    experiment_id = str(experiment_id).strip()
    if not experiment_id:
        return None

    updated: Optional[ExperimentRecord] = None
    with _LOCK:
        records = _read_all()
        for idx, record in enumerate(records):
            if record.id != experiment_id:
                continue
            payload = record.to_dict()
            payload.update({k: v for k, v in fields.items() if v is not None})
            payload["updated_at"] = _now_iso()
            updated = ExperimentRecord(
                id=str(payload.get("id", record.id)),
                title=str(payload.get("title", record.title)),
                hypothesis=str(payload.get("hypothesis", record.hypothesis)),
                status=str(payload.get("status", record.status)),
                reflection_path=payload.get("reflection_path"),
                reflection_timestamp=payload.get("reflection_timestamp"),
                arms=list(payload.get("arms") or []),
                metrics=dict(payload.get("metrics") or {}),
                outcome=payload.get("outcome"),
                notes=payload.get("notes"),
                tags=list(payload.get("tags") or []),
                created_at=str(payload.get("created_at", record.created_at)),
                updated_at=str(payload.get("updated_at")),
            )
            records[idx] = updated
            break
        _write_all(records)

    if updated is not None:
        _emit_telemetry(updated, "updated")
    return updated


def get_experiment(experiment_id: str) -> Optional[ExperimentRecord]:
    experiment_id = str(experiment_id).strip()
    if not experiment_id:
        return None
    for record in _read_all():
        if record.id == experiment_id:
            return record
    return None


__all__ = [
    "ExperimentRecord",
    "create_experiment",
    "list_experiments",
    "update_experiment",
    "get_experiment",
]
