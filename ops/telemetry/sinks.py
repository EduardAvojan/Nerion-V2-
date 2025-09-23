"""Built-in telemetry sinks."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Sequence

import sqlite3

from ops.security.fs_guard import ensure_in_repo_auto

from .bus import TelemetrySink
from .schema import TelemetryEvent


class JsonlSink(TelemetrySink):
    """Append telemetry events to `out/telemetry/events.jsonl`."""

    supports_batch = True

    def __init__(self, path: Path | str | None = None) -> None:
        target = Path(path) if path is not None else Path("out/telemetry/events.jsonl")
        self._path = Path(ensure_in_repo_auto(target))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def emit(self, event: TelemetryEvent) -> None:
        self.emit_batch([event])

    def emit_batch(self, events: Sequence[TelemetryEvent]) -> None:
        if not events:
            return
        payloads = [json.dumps(evt.to_dict(), ensure_ascii=False) for evt in events]
        data = "\n".join(payloads) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(data)


class SQLiteSink(TelemetrySink):
    """Persist telemetry events into a lightweight SQLite database."""

    supports_batch = True

    def __init__(self, path: Path | str | None = None) -> None:
        target = Path(path) if path is not None else Path("out/telemetry/events.sqlite")
        self._path = Path(ensure_in_repo_auto(target))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self._path.as_posix(), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    subject TEXT,
                    metadata TEXT,
                    payload TEXT,
                    tags TEXT,
                    redacted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_kind_ts ON telemetry_events(kind, timestamp)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_source_ts ON telemetry_events(source, timestamp)"
            )
            self._conn.commit()

    def emit(self, event: TelemetryEvent) -> None:
        self.emit_batch([event])

    def emit_batch(self, events: Sequence[TelemetryEvent]) -> None:
        if not events:
            return
        rows = []
        for evt in events:
            data = evt.to_dict()
            rows.append(
                (
                    data.get("timestamp"),
                    data.get("kind"),
                    data.get("source"),
                    data.get("subject"),
                    json.dumps(data.get("metadata") or {}, ensure_ascii=False),
                    json.dumps(data.get("payload"), ensure_ascii=False),
                    json.dumps(data.get("tags") or [], ensure_ascii=False),
                    1 if data.get("redacted") else 0,
                )
            )
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO telemetry_events (
                    timestamp, kind, source, subject, metadata, payload, tags, redacted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()


__all__ = ["JsonlSink", "SQLiteSink"]
