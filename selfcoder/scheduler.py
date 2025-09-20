from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from selfcoder.selfaudit import generate_improvement_plan
from selfcoder.plans.schema import validate_plan

def run_audit_job(root: Path, interval_seconds: int, once: bool = False) -> None:
    """
    Run self-audit job periodically. Controlled via env:
      SELFAUDIT_ENABLE=1 to enable
      SELFAUDIT_INTERVAL (seconds, default 86400 = daily)
    """
    root = Path(root)
    while True:
        try:
            plan = generate_improvement_plan(root)
            validate_plan(plan)
            # Optionally: could log or store plan path
            print(f"[scheduler] generated plan with {len(plan.get('actions', []))} actions")
        except Exception as e:
            print(f"[scheduler] error: {e}")
        if once:
            break
        time.sleep(interval_seconds)

def should_enable_scheduler() -> Optional[int]:
    if os.getenv("SELFAUDIT_ENABLE") not in {"1", "true", "yes"}:
        return None
    try:
        return int(os.getenv("SELFAUDIT_INTERVAL", "86400"))
    except Exception:
        return 86400
