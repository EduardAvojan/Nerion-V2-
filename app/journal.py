"""Application-facing journal API (façade).

This module is the stable interface the rest of the app should import:
    from app import journal

Internals are implemented in the core layer. We intentionally keep the app
layer decoupled from storage details so we can swap implementations (e.g.,
JSON → SQLite) without touching callers.

Note: the core implementation module is being renamed from
`core.memory.journal` → `core.memory.journal_store`. The import logic below
supports both during the transition.
"""

from pathlib import Path
from typing import Any, Dict, List

# Thin façade over core journal (transition-safe import)
try:
    # Preferred new path
    from core.memory import journal_store as _core  # type: ignore
except Exception:
    try:
        # Back-compat fallback (pre-rename)
        from core.memory import journal as _core  # type: ignore
    except Exception:  # pragma: no cover – fail soft to avoid breaking app flows
        _core = None  # type: ignore

def append(entry: Dict[str, Any], path: Path | str | None = None) -> None:
    """Append a journal entry, delegating to core journal.

    Backward-compat signature preserved ("path" arg ignored). We normalize the
    incoming entry to the structured core schema:
      - kind: str (default "app_log")
      - rationale: str (optional)
      - other keys go into **fields

    Best-effort: never let journal issues break main flow.
    """
    kind = str(entry.get("kind", "app_log"))
    rationale = str(entry.get("rationale", ""))
    # remove fields that core handles explicitly
    fields = {k: v for k, v in entry.items() if k not in {"kind", "rationale"}}
    try:
        if _core is not None:
            _core.log_event(kind, rationale, **fields)
    except Exception:
        # best-effort logging; never let journal issues break main flow
        pass

def tail(n: int = 20, path: Path | str | None = None) -> List[Dict[str, Any]]:
    """Return last N entries via core timeline.

    Best-effort: never let journal issues break main flow.
    """
    try:
        if _core is None:
            return []
        return list(_core.timeline(limit=max(0, n)))
    except Exception:
        return []

def by_day(day_iso: str, path: Path | str | None = None) -> List[Dict[str, Any]]:
    """Return entries whose ISO date prefix matches YYYY-MM-DD.

    Best-effort: never let journal issues break main flow.
    """
    day = (day_iso or "")[:10]
    if not day:
        return []
    try:
        if _core is None:
            return []
        rows = _core.timeline(limit=1000)
        return [e for e in rows if str(e.get("timestamp", ""))[:10] == day]
    except Exception:
        return []

def query(kind: str | None = None, since: str | None = None, until: str | None = None, contains: str | None = None, limit: int = 200) -> List[Dict[str, Any]]:
    """Thin facade over core query, best-effort."""
    try:
        if _core is None:
            return []
        return _core.query(kind=kind, since=since, until=until, contains=contains, limit=limit)
    except Exception:
        return []

def migrate_legacy() -> int:
    """Thin facade over core migrate_legacy_jsonl, best-effort."""
    try:
        if _core is None:
            return 0
        return _core.migrate_legacy_jsonl()
    except Exception:
        return 0