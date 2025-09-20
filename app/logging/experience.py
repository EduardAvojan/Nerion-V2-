

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import os
import time
import uuid
import json

DEFAULT_LOG_PATH = "out/experience/log.jsonl"
AB_CONFIG_PATH = "out/learning/ab.json"

@dataclass
class ExperienceRecord:
    event_id: str
    ts: float
    user_query: str
    parent_decision: Dict[str, Any]
    action_taken: Dict[str, Any]
    outcome_success: bool
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    network_used: Optional[bool] = None
    experiment: Optional[dict] = None

class ExperienceLogger:
    """Append-only JSONL logger for (query, decision, action, outcome).

    Each `.log(...)` call appends a single line JSON record so the notebook
    is trivially streamable for later learning and analytics.
    """
    def __init__(self, path: str = DEFAULT_LOG_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _maybe_rotate(self) -> None:
        """Rotate the JSONL log when it exceeds the max size.

        Controlled by env: NERION_LOG_ROTATE_BYTES (default: 50MB).
        Rotation renames current file to log-YYYYmmdd-HHMMSS.jsonl.
        """
        try:
            maxb_raw = os.getenv('NERION_LOG_ROTATE_BYTES') or '52428800'
            maxb = int(maxb_raw)
        except Exception:
            maxb = 52428800
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) > maxb:
                base = os.path.dirname(self.path)
                ts = time.strftime('%Y%m%d-%H%M%S')
                rotated = os.path.join(base, f'log-{ts}.jsonl')
                os.replace(self.path, rotated)
        except Exception:
            # Never crash due to rotation
            pass

    def _assign_experiment(self, user_query: str) -> Optional[dict]:
        """Return an experiment assignment dict or None if no experiment is active.

        Precedence: ab.json file -> env vars. Deterministic hashing over user_query by default.
        """
        import json as _json
        name = None
        split = None
        arms = None
        assign_by = None
        try:
            if os.path.exists(AB_CONFIG_PATH):
                with open(AB_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    cfg = _json.loads(f.read())
                name = (cfg.get('name') or '').strip() or None
                split = float(cfg.get('split', 0.5))
                arms = cfg.get('arms') or ['baseline', 'bandit+credit']
                assign_by = (cfg.get('assign_by') or 'query').strip().lower()
        except Exception:
            name = None
        if not name:
            # Fallback to environment
            env_name = os.getenv('NERION_EXPERIMENT_NAME')
            if not env_name:
                return None
            name = env_name.strip()
            try:
                split = float((os.getenv('NERION_EXPERIMENT_SPLIT') or '0.5').strip())
            except Exception:
                split = 0.5
            arms = [(os.getenv('NERION_EXPERIMENT_ARMS') or 'baseline,bandit+credit').split(',')]
            try:
                arms = [a.strip() for a in (arms[0] if isinstance(arms, list) else arms)]
            except Exception:
                arms = ['baseline', 'bandit+credit']
            assign_by = (os.getenv('NERION_EXPERIMENT_ASSIGN_BY') or 'query').strip().lower()
        # Deterministic hash
        key = user_query if assign_by == 'query' else (os.getenv('NERION_SESSION_ID') or user_query)
        try:
            import hashlib
            h = hashlib.sha1((key or '').encode('utf-8')).hexdigest()
            x = int(h[:8], 16) / 0xFFFFFFFF
        except Exception:
            x = 0.0
        arm_idx = 0 if x < max(0.0, min(1.0, split or 0.5)) else 1
        try:
            arm = arms[arm_idx] if isinstance(arms, list) and len(arms) > arm_idx else ('control' if arm_idx == 0 else 'treatment')
        except Exception:
            arm = 'control' if arm_idx == 0 else 'treatment'
        return {"name": name, "arm": arm}

    def log(
        self,
        user_query: str,
        parent_decision: Dict[str, Any],
        action_taken: Dict[str, Any],
        outcome_success: bool,
        *,
        error: Optional[str] = None,
        latency_ms: Optional[int] = None,
        network_used: Optional[bool] = None,
    ) -> str:
        rec = ExperienceRecord(
            event_id=str(uuid.uuid4()),
            ts=time.time(),
            user_query=user_query,
            parent_decision=parent_decision,
            action_taken=action_taken,
            outcome_success=outcome_success,
            error=error,
            latency_ms=latency_ms,
            network_used=network_used,
            experiment=self._assign_experiment(user_query),
        )
        # Rotate if needed before appending
        self._maybe_rotate()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return rec.event_id
