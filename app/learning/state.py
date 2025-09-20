

from __future__ import annotations
from typing import Optional, Dict, Any, List
import os
import json
import time

DEFAULT_STATE_PATH = "out/learning/state.json"

class LearningState:
    """Lightweight persistent state for the Proactive Learning Engine.

    Stores last evolution timestamp and a history of proposals so we can
    rate-limit/evaluate readiness over time without scanning the whole log.
    """

    def __init__(self, path: str = DEFAULT_STATE_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data: Dict[str, Any] = {
            "last_evolution_ts": None,   # epoch seconds
            "last_examples_count": 0,
            "last_success_rate": None,
            "last_unique_intents": 0,
            "last_proposal_ts": None,
            "proposals": [],            # list of {id, ts, n_examples, success_rate, unique_intents, audit, status}
        }
        self._load()

    # ------------------------------ persistence ------------------------------
    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._data.update(data)
        except Exception:
            # first run or unreadable file → start fresh
            pass

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ------------------------------ getters/setters --------------------------
    @property
    def last_evolution_ts(self) -> Optional[float]:
        return self._data.get("last_evolution_ts")

    def set_evolved_now(self) -> None:
        self._data["last_evolution_ts"] = time.time()

    def record_outcome_snapshot(self, *, examples: int, success_rate: Optional[float], unique_intents: int) -> None:
        self._data["last_examples_count"] = int(examples)
        self._data["last_success_rate"] = None if success_rate is None else float(success_rate)
        self._data["last_unique_intents"] = int(unique_intents)

    def record_proposal(self, proposal: Dict[str, Any]) -> None:
        self._data["last_proposal_ts"] = time.time()
        self._data.setdefault("proposals", []).append(proposal)

    # ------------------------------ helpers ----------------------------------
    def proposals(self) -> List[Dict[str, Any]]:
        val = self._data.get("proposals") or []
        return [p for p in val if isinstance(p, dict)]