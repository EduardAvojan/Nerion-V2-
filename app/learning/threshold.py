

from __future__ import annotations
from typing import Dict, Any
import os
import time

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

DEFAULTS: Dict[str, Any] = {
    "readiness": {
        "min_examples": 300,
        "min_success_rate": 0.70,
        "min_days_since_last": 14,
        "min_unique_intents": 5,
        "intent_weights": {},
    },
    "notebook_path": "out/experience/log.jsonl",
    "state_path": "out/learning/state.json",
}


def load_learning_policy(path: str = "config/learning.yaml") -> Dict[str, Any]:
    """Load proactive-learning thresholds from YAML if present; otherwise defaults.
    Returns a dict with keys: readiness, notebook_path, state_path.
    """
    cfg: Dict[str, Any] = {}
    if yaml is not None and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    cfg = data
        except Exception:
            cfg = {}

    # Merge defaults → cfg (shallow + nested readiness)
    rd = dict(DEFAULTS["readiness"])  # copy
    rd.update(cfg.get("readiness", {}) or {})
    return {
        "readiness": rd,
        "notebook_path": cfg.get("notebook_path", DEFAULTS["notebook_path"]),
        "state_path": cfg.get("state_path", DEFAULTS["state_path"]),
    }


def days_since(ts: float | None) -> float:
    """Return days elapsed since epoch ts; large sentinel if ts is None."""
    if not ts:
        return 1e9
    return (time.time() - ts) / 86400.0