"""Queryable interface over Nerion telemetry events."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ops.security.fs_guard import ensure_in_repo_auto

DEFAULT_DB_PATH = Path("out/telemetry/events.sqlite")


def _resolve_db_path(path: Optional[str | os.PathLike[str]]) -> Path:
    if path:
        return Path(ensure_in_repo_auto(Path(path)))
    env_path = os.getenv("NERION_V2_TELEMETRY_SQLITE_PATH")
    if env_path:
        return Path(ensure_in_repo_auto(Path(env_path)))
    return Path(ensure_in_repo_auto(DEFAULT_DB_PATH))


def _coerce_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


class TelemetryStore:
    """Convenience layer for querying telemetry events from SQLite."""

    def __init__(self, path: Optional[str | os.PathLike[str]] = None, *, readonly: bool = True) -> None:
        self._path = _resolve_db_path(path)
        self._readonly = bool(readonly)
        self._conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        uri_base = self._path.resolve().as_posix()
        uri = f"file:{uri_base}?mode={'ro' if self._readonly else 'rwc'}"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def ping(self) -> bool:
        try:
            with self._cursor() as cur:
                cur.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def latest_events(
        self,
        *,
        limit: int = 100,
        kind: Optional[str] = None,
        source: Optional[str] = None,
        tags_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        clauses: List[str] = []
        params: List[Any] = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if tags_contains:
            clauses.append("tags LIKE ?")
            params.append(f"%{tags_contains}%")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = (
            "SELECT timestamp, kind, source, subject, metadata, payload, tags, redacted "
            "FROM telemetry_events"
            f"{where} ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)
        with self._cursor() as cur:
            rows = cur.execute(sql, params).fetchall()
        events: List[Dict[str, Any]] = []
        for row in rows:
            events.append(_row_to_event_dict(row))
        return events

    def counts_by_kind(
        self,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, int]:
        clauses: List[str] = []
        params: List[Any] = []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if source:
            clauses.append("source = ?")
            params.append(source)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT kind, COUNT(*) AS c FROM telemetry_events{where} GROUP BY kind"
        with self._cursor() as cur:
            rows = cur.execute(sql, params).fetchall()
        return {str(row["kind"]): int(row["c"]) for row in rows}

    def counts_by_source(
        self,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> Dict[str, int]:
        clauses: List[str] = []
        params: List[Any] = []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT source, COUNT(*) AS c FROM telemetry_events{where} GROUP BY source"
        with self._cursor() as cur:
            rows = cur.execute(sql, params).fetchall()
        return {str(row["source"]): int(row["c"]) for row in rows}

    def events_between(
        self,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        kinds: Optional[Iterable[str]] = None,
        limit: int = 5000,
        descending: bool = False,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if kinds:
            kinds_list = [str(k) for k in kinds]
            placeholders = ",".join(["?"] * len(kinds_list))
            clauses.append(f"kind IN ({placeholders})")
            params.extend(kinds_list)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        order = "DESC" if descending else "ASC"
        limit = max(1, min(limit, 20000))
        sql = (
            "SELECT id, timestamp, kind, source, subject, metadata, payload, tags, redacted "
            f"FROM telemetry_events{where} ORDER BY timestamp {order} LIMIT ?"
        )
        params.append(limit)
        with self._cursor() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [_row_to_event_dict(row) for row in rows]

    def distinct_tags(self) -> List[str]:
        sql = "SELECT tags FROM telemetry_events WHERE tags IS NOT NULL AND tags <> ''"
        tags: List[str] = []
        with self._cursor() as cur:
            for row in cur.execute(sql):
                try:
                    raw = row[0]
                    items = json.loads(raw) if raw else []
                    if isinstance(items, list):
                        tags.extend(str(x) for x in items)
                except Exception:
                    continue
        return sorted({tag for tag in tags if tag})

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def _row_to_event_dict(row: sqlite3.Row) -> Dict[str, Any]:
    metadata = _safe_json_load(row["metadata"]) if row["metadata"] else {}
    payload = _safe_json_load(row["payload"]) if row["payload"] else None
    tags = _safe_json_load(row["tags"]) if row["tags"] else []
    ts = _coerce_dt(row["timestamp"])
    return {
        "id": row["id"] if "id" in row.keys() else None,
        "timestamp": ts.isoformat().replace("+00:00", "Z") if ts else row["timestamp"],
        "kind": row["kind"],
        "source": row["source"],
        "subject": row["subject"],
        "metadata": metadata,
        "payload": payload,
        "tags": tags,
        "redacted": bool(row["redacted"]),
    }


def _safe_json_load(raw: Any) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


__all__ = ["TelemetryStore"]
