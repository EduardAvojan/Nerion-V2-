from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.security import fs_guard


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class SessionCache:
    SCHEMA_VERSION = 1

    def __init__(self, path: str, ns: Dict[str, str], max_turns: int = 50) -> None:
        safe = fs_guard.ensure_in_repo(Path('.'), path)
        self.path = Path(safe)
        self.ns = ns
        self.max_turns = max_turns
        self.state: Dict[str, Any] = {
            "schema": self.SCHEMA_VERSION,
            "ns": ns,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "turns": [],
            "last_ref": None,
            "short_facts": [],
            "summary": None,
        }

    # --- persistence -----------------------------------------------------
    def _atomic_write(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".session.", dir=self.path.parent)
        try:
            with io.open(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    def save(self) -> None:
        self.state['turns'] = self.state.get('turns', [])[-self.max_turns:]
        self.state['updated_at'] = _now_iso()
        self._atomic_write(self.state)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if data.get('ns') != self.ns:
            return
        if int(data.get('schema', 0)) != self.SCHEMA_VERSION:
            return
        self.state = data

    # --- mutation helpers -------------------------------------------------
    def record_turn(self, role: str, text: str) -> None:
        turns = self.state.setdefault('turns', [])
        turns.append({"role": role, "text": text, "ts": _now_iso()})

    def set_last_ref(self, value: Optional[str]) -> None:
        if value:
            self.state['last_ref'] = {"value": value, "ts": _now_iso()}

    def upsert_short_fact(self, fact: str, tags: List[str], score: float, ttl_days: Optional[int]) -> Dict[str, Any]:
        facts = self.state.setdefault('short_facts', [])
        key = fact.strip().lower()
        for item in facts:
            if item.get('fact', '').strip().lower() == key:
                item['score'] = min(5.0, float(item.get('score', 0.0)) + score)
                item['tags'] = sorted(list(set((item.get('tags') or []) + tags)))
                if ttl_days is not None:
                    item['ttl_days'] = ttl_days
                item['ts'] = _now_iso()
                item['count'] = int(item.get('count', 0)) + 1
                return item
        new_item = {
            'fact': fact,
            'tags': tags,
            'score': min(5.0, score),
            'ts': _now_iso(),
            'ttl_days': ttl_days,
            'count': 1,
        }
        facts.append(new_item)
        return new_item

    def decay_and_prune(self, decay_per_day: float, default_ttl_days: int, promotion_threshold: float) -> List[Dict[str, Any]]:
        now = datetime.utcnow()
        keep: List[Dict[str, Any]] = []
        promos: List[Dict[str, Any]] = []
        for item in self.state.get('short_facts', []):
            try:
                ts = datetime.fromisoformat(item.get('ts', '').replace('Z', ''))
            except Exception:
                ts = now
            days = max(0.0, (now - ts).total_seconds() / 86400.0)
            ttl = item.get('ttl_days') or default_ttl_days
            if days >= ttl:
                continue
            item['score'] = max(0.0, float(item.get('score', 0.0)) - decay_per_day * days)
            item['ts'] = _now_iso()
            keep.append(item)
            if item['score'] >= promotion_threshold:
                promos.append(item)
        self.state['short_facts'] = keep
        return promos

    def update_summary(self, turns: List[Dict[str, Any]]) -> None:
        snippets = []
        for turn in turns[-10:]:
            role = turn.get('role')
            text = (turn.get('text') or '').strip()
            if role == 'user' and text:
                snippets.append(text)
        if snippets:
            self.state['summary'] = 'Recent discussion: ' + ' | '.join(snippets[:3])
