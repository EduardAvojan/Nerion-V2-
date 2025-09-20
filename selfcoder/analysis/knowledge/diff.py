

from __future__ import annotations

import difflib
import hashlib
from typing import Any, Dict


def text_hash(text: str) -> str:
    """Stable hash for text to track duplicates."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def diff_artifacts(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a lightweight diff between two artifacts.

    Returns a dict with:
      - overlap: ratio of shared lines
      - additions: new lines
      - deletions: removed lines
      - contradicted: heuristic flag if conclusions contradict
    """
    old_text = (old.get("summary") or "") + " " + (old.get("conclusion") or "")
    new_text = (new.get("summary") or "") + " " + (new.get("conclusion") or "")

    old_lines = [line.strip() for line in old_text.splitlines() if line.strip()]
    new_lines = [line.strip() for line in new_text.splitlines() if line.strip()]

    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    overlap = sm.ratio()

    diff = {
        "overlap": overlap,
    "additions": [line for line in new_lines if line not in old_lines],
    "deletions": [line for line in old_lines if line not in new_lines],
        "contradicted": False,
    }

    # crude contradiction check: if both have conclusions and they differ significantly
    old_conc = (old.get("conclusion") or "").lower()
    new_conc = (new.get("conclusion") or "").lower()
    if old_conc and new_conc and old_conc != new_conc:
        diff["contradicted"] = True

    return diff


def adjust_weights(profile: Dict[str, Any], diff: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust profile weights based on new evidence, with safe bounds.

    We keep a small, transparent nudging policy and clamp
    recency/authority/agreement to [0.05, 0.6] to avoid runaway effects.
    """
    # Ensure weights exist with sensible defaults
    weights = dict(profile.get("weights", {}))
    weights.setdefault("recency", 0.30)
    weights.setdefault("authority", 0.30)
    weights.setdefault("agreement", 0.40)

    # Apply tiny nudges
    if diff.get("contradicted"):
        # If contradicted later, reduce reliance on recency a bit
        weights["recency"] = weights.get("recency", 0.30) - 0.05
    else:
        # If overlap is high, increase recency slightly
        if float(diff.get("overlap", 0.0)) > 0.70:
            weights["recency"] = weights.get("recency", 0.30) + 0.05

    # Clamp all to [0.05, 0.6]
    for k in ("recency", "authority", "agreement"):
        v = float(weights.get(k, 0.30))
        weights[k] = max(0.05, min(0.60, v))

    profile["weights"] = weights
    return profile


def track_calibration(profile: Dict[str, Any], diff: Dict[str, Any], artifact_id: str) -> None:
    """Record calibration info for future tuning."""
    cal = profile.setdefault("calibration", [])
    cal.append(
        {
            "artifact_id": artifact_id,
            "overlap": diff.get("overlap"),
            "contradicted": diff.get("contradicted"),
        }
    )