"""Policy-aware prioritisation for architect briefs.

This module scores architect briefs against policy thresholds so autonomous
routines can decide which upgrades may run automatically ("auto"), which need
human review ("review"), and which should be blocked entirely ("block").

It also exposes helpers to supply the most relevant brief context to the
planning pipeline so heuristics/LLM planners inherit telemetry-backed signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
import os
import math

from selfcoder.planner.architect_briefs import ArchitectBrief, generate_architect_briefs

try:  # pragma: no cover - defensive import
    from selfcoder.config import get_policy as _get_policy
except Exception:  # pragma: no cover
    def _get_policy(default: str = "balanced") -> str:
        return default


# Policy thresholds tune how aggressive Nerion can be when self-upgrading.
# Values are calibrated so "safe" errs on human review, "fast" favours
# autonomous execution, and "balanced" sits in the middle.
_POLICY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "safe": {
        "review_risk": 4.0,
        "block_risk": 7.5,
        "review_effort": 3.0,
        "block_effort": 5.5,
        "test_failures_review": 1.0,
        "apply_failures_review": 1.0,
        "apply_failures_block": 2.0,
        "cost_multiplier": 80.0,
        "default_cost_budget": 400.0,
    },
    "balanced": {
        "review_risk": 6.0,
        "block_risk": 10.0,
        "review_effort": 5.0,
        "block_effort": 8.5,
        "test_failures_review": 2.0,
        "apply_failures_review": 1.0,
        "apply_failures_block": 3.0,
        "cost_multiplier": 70.0,
        "default_cost_budget": 600.0,
    },
    "fast": {
        "review_risk": 8.0,
        "block_risk": 12.0,
        "review_effort": 6.5,
        "block_effort": 10.0,
        "test_failures_review": 3.0,
        "apply_failures_review": 2.0,
        "apply_failures_block": 4.0,
        "cost_multiplier": 60.0,
        "default_cost_budget": 800.0,
    },
}

_DECISION_ORDER = {"auto": 0, "review": 1, "block": 2}


def resolve_policy(default: str = "balanced") -> str:
    """Public helper to resolve the effective runtime policy."""

    return _policy_for_runtime(default)


def get_policy_thresholds(policy: str) -> Dict[str, float]:
    """Expose policy threshold table for other gating modules (e.g., apply)."""

    return dict(_thresholds_for_policy(policy))


@dataclass(slots=True)
class PrioritizedBrief:
    """Wraps an ArchitectBrief with policy-aware metadata."""

    brief: ArchitectBrief
    decision: str
    reasons: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    effort_score: float = 0.0
    estimated_cost: float = 0.0
    effective_priority: float = 0.0
    policy: str = "balanced"
    gating: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - thin wrapper
        data = self.brief.to_dict()
        data.update(
            {
                "decision": self.decision,
                "reasons": list(self.reasons),
                "risk_score": float(self.risk_score),
                "effort_score": float(self.effort_score),
                "estimated_cost": float(self.estimated_cost),
                "effective_priority": float(self.effective_priority),
                "policy": self.policy,
                "gating": dict(self.gating),
            }
        )
        return data


def _policy_for_runtime(default: str = "balanced") -> str:
    try:
        policy = _get_policy(default)
    except Exception:
        policy = default
    return (policy or default).strip().lower() or default


def _thresholds_for_policy(policy: str) -> Dict[str, float]:
    return _POLICY_THRESHOLDS.get(policy, _POLICY_THRESHOLDS["balanced"])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _collect_signal_metrics(brief: ArchitectBrief) -> Dict[str, Any]:
    signals = brief.signals or {}
    hotspot = signals.get("hotspot") or {}
    coverage = signals.get("coverage") or {}
    smells = signals.get("smells") or {}

    return {
        "risk_score": _safe_float(hotspot.get("risk_score")),
        "test_failures": _safe_int(hotspot.get("test_failures")),
        "apply_failures": _safe_int(hotspot.get("apply_failures")),
        "recent_fix_commits": _safe_int(hotspot.get("recent_fix_commits")),
        "missing_lines": _safe_int(coverage.get("missing_lines")),
        "coverage_files": list(set(_safe_string_list(coverage.get("files")))),
        "smell_count": _safe_int(smells.get("count")),
        "smell_examples": _safe_dict_list(smells.get("examples")),
        "component": brief.component,
    }


def _safe_string_list(value: Any) -> List[str]:
    strings: List[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str) and item:
                strings.append(item)
    elif isinstance(value, str) and value:
        strings.append(value)
    return strings


def _safe_dict_list(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                out.append(item)
    return out


def _compute_effort(metrics: Dict[str, Any]) -> float:
    missing = metrics.get("missing_lines", 0)
    smell_count = metrics.get("smell_count", 0)
    recent_fixes = metrics.get("recent_fix_commits", 0)
    coverage_files = len(metrics.get("coverage_files", []))
    # Normalise effort so "1" roughly equals a small patch worth of work
    effort = (missing / 15.0) + (smell_count * 0.6) + (recent_fixes * 0.2) + (coverage_files * 0.3)
    return round(effort, 2)


def _compute_risk(metrics: Dict[str, Any]) -> float:
    base = metrics.get("risk_score", 0.0)
    test_failures = metrics.get("test_failures", 0)
    apply_failures = metrics.get("apply_failures", 0)
    boost = (test_failures * 1.8) + (apply_failures * 2.5)
    return round(base + boost, 2)


def _estimate_cost(effort: float, thresholds: Dict[str, float]) -> float:
    multiplier = thresholds.get("cost_multiplier", 70.0)
    return round(max(effort, 0.0) * multiplier, 2)


def _default_cost_budget(policy: str, thresholds: Dict[str, float]) -> Optional[float]:
    env_budget = os.getenv("NERION_ARCHITECT_COST_BUDGET")
    if env_budget:
        try:
            return float(env_budget)
        except Exception:
            pass
    return thresholds.get("default_cost_budget")


def prioritize_briefs(
    briefs: Sequence[ArchitectBrief],
    *,
    policy: Optional[str] = None,
    cost_budget: Optional[float] = None,
) -> List[PrioritizedBrief]:
    """Score briefs and apply policy gating decisions."""

    if not briefs:
        return []

    pol = (policy or _policy_for_runtime()).lower()
    thresholds = _thresholds_for_policy(pol)
    budget = cost_budget
    if budget is None:
        budget = _default_cost_budget(pol, thresholds)

    prioritized: List[PrioritizedBrief] = []
    for brief in briefs:
        metrics = _collect_signal_metrics(brief)
        risk = _compute_risk(metrics)
        effort = _compute_effort(metrics)
        est_cost = _estimate_cost(effort, thresholds)

        decision = "auto"
        reasons: List[str] = []
        gating: Dict[str, Any] = {
            "risk": {
                "score": risk,
                "review": thresholds.get("review_risk"),
                "block": thresholds.get("block_risk"),
            },
            "effort": {
                "score": effort,
                "review": thresholds.get("review_effort"),
                "block": thresholds.get("block_effort"),
            },
            "cost": {
                "estimate": est_cost,
                "budget": budget,
            },
        }

        apply_failures = metrics.get("apply_failures", 0)
        test_failures = metrics.get("test_failures", 0)

        if apply_failures >= thresholds.get("apply_failures_block", math.inf):
            decision = "block"
            reasons.append(
                f"Apply failures {apply_failures} exceed block threshold {thresholds.get('apply_failures_block')}"
            )
        elif risk >= thresholds.get("block_risk", math.inf):
            decision = "block"
            reasons.append(
                f"Risk score {risk:.1f} >= block threshold {thresholds.get('block_risk')}"
            )
        else:
            if apply_failures >= thresholds.get("apply_failures_review", math.inf):
                decision = "review"
                reasons.append(
                    f"Apply failures {apply_failures} require review (policy {pol})"
                )
            if risk >= thresholds.get("review_risk", math.inf):
                decision = "review"
                reasons.append(
                    f"Risk score {risk:.1f} exceeds review threshold {thresholds.get('review_risk')}"
                )
            if test_failures >= thresholds.get("test_failures_review", math.inf):
                decision = "review"
                reasons.append(
                    f"Test failures {test_failures} exceed review threshold {thresholds.get('test_failures_review')}"
                )
            if effort >= thresholds.get("block_effort", math.inf):
                decision = "block"
                reasons.append(
                    f"Effort {effort:.1f} >= block effort threshold {thresholds.get('block_effort')}"
                )
            elif effort >= thresholds.get("review_effort", math.inf) and decision != "block":
                decision = "review"
                reasons.append(
                    f"Effort {effort:.1f} exceeds review threshold {thresholds.get('review_effort')}"
                )
            if budget is not None and est_cost > budget and decision != "block":
                decision = "review"
                reasons.append(
                    f"Estimated cost ${est_cost:.0f} exceeds budget ${budget:.0f}"
                )

        effective_pri = round(brief.priority + (risk * 0.35) + (effort * 0.2), 2)
        prioritized.append(
            PrioritizedBrief(
                brief=brief,
                decision=decision,
                reasons=reasons,
                risk_score=risk,
                effort_score=effort,
                estimated_cost=est_cost,
                effective_priority=effective_pri,
                policy=pol,
                gating=gating,
            )
        )

    prioritized.sort(
        key=lambda item: (
            _DECISION_ORDER.get(item.decision, 99),
            -item.effective_priority,
            -item.risk_score,
        )
    )
    return prioritized


def _component_hint_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    path = path.strip()
    if not path:
        return None
    path = path.replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return None
    return parts[0]


def _component_hint_from_instruction(instruction: str, prioritized: Sequence[PrioritizedBrief]) -> Optional[str]:
    if not instruction:
        return None
    lower = instruction.lower()
    for item in prioritized:
        comp = (item.brief.component or "").lower()
        if comp and comp in lower:
            return item.brief.component
    return None


def _suggested_targets(brief: PrioritizedBrief) -> List[str]:
    metrics = _collect_signal_metrics(brief.brief)
    candidates: List[str] = []
    signals = brief.brief.signals or {}
    coverage_block = signals.get("coverage") if isinstance(signals, dict) else {}
    if not isinstance(coverage_block, dict):
        coverage_block = {}
    example = coverage_block.get("example_file")
    if isinstance(example, str) and example:
        candidates.append(example)
    for path in metrics.get("coverage_files", []):
        if path and path not in candidates:
            candidates.append(path)
    for smell in metrics.get("smell_examples", []):
        path = smell.get("path") if isinstance(smell, dict) else None
        if isinstance(path, str) and path and path not in candidates:
            candidates.append(path)
    # Fall back to component root hint
    component = metrics.get("component")
    if component and component not in candidates:
        candidates.append(component)
    return candidates[:5]


def _serialize_alternates(items: Sequence[PrioritizedBrief], primary_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    alternates: List[Dict[str, Any]] = []
    for item in items:
        if item.brief.id == primary_id:
            continue
        alternates.append(
            {
                "id": item.brief.id,
                "component": item.brief.component,
                "decision": item.decision,
                "effective_priority": item.effective_priority,
                "risk_score": item.risk_score,
            }
        )
        if len(alternates) >= limit:
            break
    return alternates


def build_planner_context(
    instruction: str,
    target_file: Optional[str] = None,
    *,
    max_briefs: int = 5,
    telemetry_window_hours: int = 48,
    policy: Optional[str] = None,
    cost_budget: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Return the most relevant prioritized brief for the given plan request."""

    try:
        briefs = generate_architect_briefs(
            max_briefs=max_briefs,
            telemetry_window_hours=telemetry_window_hours,
        )
    except Exception:
        return None

    prioritized = prioritize_briefs(briefs, policy=policy, cost_budget=cost_budget)
    if not prioritized:
        return None

    component_hint = _component_hint_from_path(target_file)
    if not component_hint:
        component_hint = _component_hint_from_instruction(instruction, prioritized)

    selected: Optional[PrioritizedBrief] = None
    if component_hint:
        for item in prioritized:
            if item.brief.component == component_hint:
                selected = item
                break
    if selected is None:
        selected = prioritized[0]

    brief_dict = selected.brief.to_dict()
    context = {
        "brief": brief_dict,
        "decision": selected.decision,
        "reasons": list(selected.reasons),
        "policy": selected.policy,
        "risk_score": selected.risk_score,
        "effort_score": selected.effort_score,
        "estimated_cost": selected.estimated_cost,
        "effective_priority": selected.effective_priority,
        "gating": dict(selected.gating),
        "suggested_targets": _suggested_targets(selected),
        "alternates": _serialize_alternates(prioritized, brief_dict.get("id", "")),
    }
    return context


__all__ = [
    "PrioritizedBrief",
    "prioritize_briefs",
    "build_planner_context",
]
