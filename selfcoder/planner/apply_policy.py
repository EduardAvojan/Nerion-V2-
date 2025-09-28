"""Apply-time policy enforcement for autonomous upgrades.

This module consumes planner metadata (e.g., architect briefs) plus runtime
policy thresholds to decide whether a plan may auto-apply, requires human
review, or must be blocked entirely.  The logic mirrors the policy-aware
prioritiser so the same knobs (safe/balanced/fast) apply consistently across
planning and execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from selfcoder.config import get_policy
from selfcoder.planner.prioritizer import (
    get_policy_thresholds,
    resolve_policy,
)


_VALID_DECISIONS = {"auto", "review", "block"}


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _merge_reasons(existing: List[str], extra: Iterable[Any]) -> None:
    for item in extra:
        if not item:
            continue
        text = str(item).strip()
        if not text:
            continue
        if text not in existing:
            existing.append(text)


@dataclass
class ApplyPolicyDecision:
    """Container describing whether a plan may auto-apply under current policy."""

    decision: str
    reasons: List[str] = field(default_factory=list)
    policy: str = "balanced"
    gating: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_blocked(self) -> bool:
        return self.decision == "block"

    def requires_manual_review(self) -> bool:
        return self.decision == "review"


def evaluate_apply_policy(
    plan: Dict[str, Any],
    *,
    policy: Optional[str] = None,
    default_decision: str = "review",
) -> ApplyPolicyDecision:
    """Return the apply gating decision for *plan* under the selected policy.

    - Falls back to planner metadata (architect brief) when available.
    - When no metadata exists, defaults to "review" so human approval is
      required unless an explicit override is provided.
    """

    metadata = plan.get("metadata") if isinstance(plan, dict) else None
    if not isinstance(metadata, dict):
        metadata = {}
    brief_meta = metadata.get("architect_brief")
    if not isinstance(brief_meta, dict):
        brief_meta = {}

    resolved_policy = (
        _coerce_str(policy)
        or _coerce_str(brief_meta.get("policy"))
        or resolve_policy()
        or get_policy()
        or "balanced"
    ).lower()
    thresholds = get_policy_thresholds(resolved_policy)

    decision = _coerce_str(brief_meta.get("decision"))
    if decision not in _VALID_DECISIONS:
        decision = default_decision

    reasons: List[str] = []
    _merge_reasons(reasons, brief_meta.get("reasons") or [])

    gating: Dict[str, Any] = {}
    gating_raw = brief_meta.get("gating")
    if isinstance(gating_raw, dict):
        gating = dict(gating_raw)

    # Risk/effort/cost gates mirror prioritizer scoring
    risk_block = _coerce_float((gating.get("risk") or {}).get("block") if isinstance(gating.get("risk"), dict) else None)
    risk_review = _coerce_float((gating.get("risk") or {}).get("review") if isinstance(gating.get("risk"), dict) else None)
    risk_score = _coerce_float((gating.get("risk") or {}).get("score") if isinstance(gating.get("risk"), dict) else None)
    if risk_block is None:
        risk_block = thresholds.get("block_risk")
    if risk_review is None:
        risk_review = thresholds.get("review_risk")
    if risk_score is None:
        risk_score = _coerce_float(brief_meta.get("risk_score"))
    if risk_score is not None:
        if risk_block is not None and risk_score >= risk_block:
            if decision != "block":
                reasons.append(f"Risk {risk_score:.1f} >= block threshold {risk_block:.1f}")
            decision = "block"
        elif risk_review is not None and risk_score >= risk_review:
            if decision == "auto":
                decision = "review"
            reasons.append(f"Risk {risk_score:.1f} exceeds review threshold {risk_review:.1f}")

    effort_block = _coerce_float((gating.get("effort") or {}).get("block") if isinstance(gating.get("effort"), dict) else None)
    effort_review = _coerce_float((gating.get("effort") or {}).get("review") if isinstance(gating.get("effort"), dict) else None)
    effort_score = _coerce_float((gating.get("effort") or {}).get("score") if isinstance(gating.get("effort"), dict) else None)
    if effort_block is None:
        effort_block = thresholds.get("block_effort")
    if effort_review is None:
        effort_review = thresholds.get("review_effort")
    if effort_score is None:
        effort_score = _coerce_float(brief_meta.get("effort_score"))
    if effort_score is not None:
        if effort_block is not None and effort_score >= effort_block:
            if decision != "block":
                reasons.append(f"Effort {effort_score:.1f} >= block threshold {effort_block:.1f}")
            decision = "block"
        elif effort_review is not None and effort_score >= effort_review:
            if decision == "auto":
                decision = "review"
            reasons.append(f"Effort {effort_score:.1f} exceeds review threshold {effort_review:.1f}")

    cost_meta = gating.get("cost") if isinstance(gating.get("cost"), dict) else {}
    cost_estimate = _coerce_float(cost_meta.get("estimate"))
    if cost_estimate is None:
        cost_estimate = _coerce_float(brief_meta.get("estimated_cost"))
    cost_budget = _coerce_float(cost_meta.get("budget"))
    if cost_budget is None:
        cost_budget = _coerce_float(thresholds.get("default_cost_budget"))
    if cost_estimate is not None and cost_budget is not None and cost_estimate > cost_budget:
        if decision != "block":
            if decision == "auto":
                decision = "review"
            reasons.append(
                f"Estimated cost ${cost_estimate:,.0f} exceeds budget ${cost_budget:,.0f}"
            )

    # Provide an explicit reason when no architect brief metadata is present.
    if not brief_meta:
        reasons.append("No architect brief metadata – defaulting to manual review")

    return ApplyPolicyDecision(
        decision=decision,
        reasons=reasons,
        policy=resolved_policy,
        gating=gating,
        metadata=brief_meta,
    )


def apply_allowed(
    decision: ApplyPolicyDecision,
    *,
    allow_review: bool = False,
    force: bool = False,
) -> bool:
    """Return whether an apply operation should proceed under gating."""

    if force:
        return True
    if decision.is_blocked():
        return False
    if decision.requires_manual_review():
        return allow_review
    return True
