"""Dataset builder for optional local LoRA fine-tuning (stub).

Builds a simple JSONL dataset from the experience log:
  {"input": <user_query>, "intent": <intent>, "tools": [..], "success": bool}

No network calls; safe offline. This is intentionally light and serves as a
starting point for local adapters.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import re

LOG_PATH = Path('out/experience/log.jsonl')


def _since_to_ts(spec: str | None) -> float | None:
    if not spec:
        return None
    s = str(spec).strip().lower()
    # Accept Nd (days)
    m = re.match(r"^(\d+)d$", s)
    if m:
        import time
        days = int(m.group(1))
        return time.time() - days * 86400.0
    # ISO datetime not implemented; return None
    return None


def build_dataset(*, since: str | None = None, domain: str | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds: List[Dict[str, Any]] = []
    if not LOG_PATH.exists():
        return ds, {"count": 0}
    since_ts = _since_to_ts(since)
    ok = 0
    total = 0
    for ln in LOG_PATH.read_text(encoding='utf-8', errors='ignore').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        ts = rec.get('ts')
        # Apply since filter only if ts looks like a UNIX epoch in seconds
        try:
            tsf = float(ts) if ts is not None else None
        except Exception:
            tsf = None
        if since_ts and tsf is not None and tsf >= 10_000_000 and tsf <= since_ts:
            continue
        intent = ((rec.get('parent_decision') or {}).get('intent') or None)
        dom = (domain or '').strip().lower()
        # Treat 'all' (default) as no filter
        if dom and dom != 'all' and intent and not str(intent).startswith(domain):
            continue
        steps = ((rec.get('action_taken') or {}).get('steps') or [])
        tools = [s.get('tool') for s in steps if isinstance(s, dict) and s.get('tool')]
        item = {
            'input': rec.get('user_query') or '',
            'intent': intent,
            'tools': tools,
            'success': bool(rec.get('outcome_success')),
        }
        ds.append(item)
    total += 1
    ok += (1 if item['success'] else 0)
    return ds, {"count": total, "successes": ok}
