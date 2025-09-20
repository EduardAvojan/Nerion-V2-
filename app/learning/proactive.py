

from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple
import os
import json
import uuid

from .threshold import load_learning_policy, days_since
from .state import LearningState

# ---------------- Notebook reader (JSONL) ----------------

def iter_experiences(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSONL notebook records one by one. Returns empty iterator if missing."""
    if not os.path.exists(path):
        return []
    def _gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    return _gen()


def summarize_notebook(path: str, since_ts: float | None) -> Dict[str, Any]:
    """Compute counts/success rate/intent histogram since a timestamp."""
    total = 0
    success = 0
    intents: Dict[str, int] = {}
    for rec in iter_experiences(path):
        ts = rec.get("ts")
        if since_ts and ts and ts <= since_ts:
            continue
        total += 1
        if rec.get("outcome_success"):
            success += 1
        dec = rec.get("parent_decision") or {}
        intent = dec.get("intent") or "unknown"
        intents[intent] = intents.get(intent, 0) + 1
    sr = (success / total) if total > 0 else 0.0
    return {"count": total, "success_rate": sr, "intents": intents}


# ---------------- Readiness heuristic ----------------

def is_ready(summary: Dict[str, Any], state: LearningState, policy: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    rd = policy["readiness"]
    min_examples = int(rd.get("min_examples", 300))
    min_sr = float(rd.get("min_success_rate", 0.70))
    min_days = float(rd.get("min_days_since_last", 14))
    min_ui = int(rd.get("min_unique_intents", 5))
    weights = rd.get("intent_weights") or {}

    # Weighted example tally if provided
    weighted = 0.0
    for k, c in (summary["intents"] or {}).items():
        weighted += float(weights.get(k, 1.0)) * c

    unique = len(summary["intents"] or {})
    days_ok = days_since(state.last_evolution_ts) >= min_days
    examples_ok = (weighted if weights else summary["count"]) >= min_examples
    sr_ok = summary["success_rate"] >= min_sr
    ui_ok = unique >= min_ui

    audit = {
        "days_since_last": days_since(state.last_evolution_ts),
        "examples": summary["count"],
        "weighted_examples": weighted,
        "success_rate": round(summary["success_rate"], 3),
        "unique_intents": unique,
        "checks": {
            "days_ok": days_ok,
            "examples_ok": examples_ok,
            "success_ok": sr_ok,
            "unique_ok": ui_ok,
        },
    }
    ok = all(audit["checks"].values())
    return ok, audit


# ---------------- Proposal builder ----------------

def build_proposal(summary: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "ts": None,
        "n_examples": summary["count"],
        "success_rate": summary["success_rate"],
        "unique_intents": len(summary["intents"] or {}),
        "audit": audit,
        "status": "proposed",
        "headline": "Evolution available",
        "message": "I have collected enough new experience data to begin a self-tuning session.",
        "options": ["now", "tonight", "remind_later"],
    }


# ---------------- Public API ----------------

def evaluate_readiness(policy_path: str = "config/learning.yaml") -> Dict[str, Any]:
    """Summarize Notebook since last evolution and decide if a proposal should be emitted.
    Returns a dict: {ready, summary, audit, state_path, proposal?}
    """
    policy = load_learning_policy(policy_path)
    state = LearningState(policy["state_path"])
    nb = summarize_notebook(policy["notebook_path"], state.last_evolution_ts)
    ok, audit = is_ready(nb, state, policy)
    result: Dict[str, Any] = {"ready": ok, "summary": nb, "audit": audit, "state_path": state.path}
    if ok:
        prop = build_proposal(nb, audit)
        state.record_proposal(prop)
        state.save()
        result["proposal"] = prop
    return result