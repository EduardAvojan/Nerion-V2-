from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any


STATE_PATH = Path("out/policies/upgrade_state.json")


def _load_state() -> Dict[str, Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}


def _save_state(st: Dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def should_shadow() -> bool:
    """Return True when shadow replay should run after a turn.

    Controlled by env `NERION_UPGRADE_SHADOW` (default on).
    """
    try:
        v = (os.getenv("NERION_UPGRADE_SHADOW") or "1").strip().lower()
        return v in {"1", "true", "yes", "on"}
    except Exception:
        return True


def _append_shadow_metrics(metrics: Dict[str, Any]) -> None:
    st = _load_state()
    log = st.setdefault("shadow", [])
    log.append({"ts": int(time.time()), "metrics": metrics})
    _save_state(st)


def run_shadow_replay() -> None:
    """Run a lightweight shadow evaluation and record metrics.

    This does not affect live execution. It generates a self-audit improvement
    plan and records timing/summary to upgrade_state.json.
    """
    t0 = time.time()
    ok = True
    try:
        from pathlib import Path as _P
        from selfcoder.selfaudit import generate_improvement_plan
        plan = generate_improvement_plan(_P("."))
        files = []
        try:
            acts = plan.get("actions") or []
            for a in acts:
                p = ((a or {}).get("payload") or {}).get("path") or ((a or {}).get("payload") or {}).get("target")
                if p:
                    files.append(str(p))
        except Exception:
            files = []
    except Exception:
        ok = False
        files = []
    dt = int((time.time() - t0) * 1000)
    metrics = {"ok": bool(ok), "latency_ms": dt, "files_considered": len(files)}
    _append_shadow_metrics(metrics)


def schedule_shadow_replay() -> None:
    try:
        th = threading.Thread(target=run_shadow_replay, name="nerion_shadow_replay", daemon=True)
        th.start()
    except Exception:
        pass

