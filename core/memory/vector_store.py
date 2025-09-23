"""Persistent vector store backed by SQLite with cosine similarity search."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ops.security.fs_guard import ensure_in_repo_auto

DEFAULT_DB_PATH = Path("out/memory/vector_store.sqlite")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_path(path: Optional[str | os.PathLike[str]]) -> Path:
    if path:
        return Path(ensure_in_repo_auto(Path(path)))
    env_path = os.getenv("NERION_VECTOR_STORE_PATH")
    if env_path:
        return Path(ensure_in_repo_auto(Path(env_path)))
    return Path(ensure_in_repo_auto(DEFAULT_DB_PATH))


def _normalize_embedding(vec: Sequence[float]) -> List[float]:
    floats = [float(x) for x in vec]
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    return [v / norm for v in floats]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    length = min(len(vec_a), len(vec_b))
    dot = sum(vec_a[i] * vec_b[i] for i in range(length))
    return max(-1.0, min(1.0, dot))


@dataclass
class VectorRecord:
    id: str
    namespace: str
    vector: List[float]
    text: Optional[str]
    metadata: Dict[str, Any]
    tags: List[str]
    score: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VectorStore:
    """Simple namespaced vector store with cosine similarity search."""

    def __init__(self, path: Optional[str | os.PathLike[str]] = None) -> None:
        self._path = _resolve_path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self._path.as_posix(), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    text TEXT,
                    metadata TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_namespace ON vectors(namespace)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    def add(
        self,
        *,
        namespace: str,
        embedding: Sequence[float],
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        item_id = item_id or uuid.uuid4().hex
        vector = _normalize_embedding(embedding)
        ts = _now_iso()
        row = (
            item_id,
            namespace,
            json.dumps(vector, ensure_ascii=False),
            text,
            json.dumps(metadata or {}, ensure_ascii=False),
            json.dumps(list(tags or []), ensure_ascii=False),
            ts,
            ts,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO vectors (id, namespace, vector, text, metadata, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            self._conn.commit()
        return item_id

    def add_many(
        self,
        records: Iterable[Tuple[Optional[str], Sequence[float], Optional[str], Optional[Dict[str, Any]], Optional[Iterable[str]]]],
        *,
        namespace: str,
    ) -> List[str]:
        ids: List[str] = []
        rows = []
        ts = _now_iso()
        for maybe_id, embedding, text, metadata, tags in records:
            item_id = maybe_id or uuid.uuid4().hex
            ids.append(item_id)
            rows.append(
                (
                    item_id,
                    namespace,
                    json.dumps(_normalize_embedding(embedding), ensure_ascii=False),
                    text,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    json.dumps(list(tags or []), ensure_ascii=False),
                    ts,
                    ts,
                )
            )
        if not rows:
            return ids
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO vectors (id, namespace, vector, text, metadata, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
        return ids

    # ------------------------------------------------------------------
    def delete(self, item_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM vectors WHERE id = ?", (item_id,))
            self._conn.commit()
            return cur.rowcount

    def delete_namespace(self, namespace: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM vectors WHERE namespace = ?", (namespace,))
            self._conn.commit()
            return cur.rowcount

    def list_namespaces(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT namespace FROM vectors ORDER BY namespace").fetchall()
        return [str(row["namespace"]) for row in rows]

    def fetch(self, item_id: str) -> Optional[VectorRecord]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM vectors WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return None
        return _row_to_record(row)

    def search(
        self,
        *,
        namespace: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[VectorRecord]:
        top_k = max(1, min(top_k, 50))
        q = _normalize_embedding(query_embedding)
        with self._lock:
            rows = self._conn.execute("SELECT * FROM vectors WHERE namespace = ?", (namespace,)).fetchall()
        scored: List[VectorRecord] = []
        for row in rows:
            record = _row_to_record(row)
            score = _cosine_similarity(q, record.vector)
            if score >= min_score:
                record.score = score
                scored.append(record)
        scored.sort(key=lambda r: r.score or 0.0, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def _row_to_record(row: sqlite3.Row) -> VectorRecord:
    vector = json.loads(row["vector"])
    metadata = json.loads(row["metadata"] or "{}")
    tags = json.loads(row["tags"] or "[]")
    return VectorRecord(
        id=row["id"],
        namespace=row["namespace"],
        vector=list(vector or []),
        text=row["text"],
        metadata=dict(metadata or {}),
        tags=list(tags or []),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


__all__ = ["VectorStore", "VectorRecord"]
