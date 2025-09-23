from __future__ import annotations

from typing import List

import pytest

from selfcoder.planner.architect_briefs import ArchitectBrief
from selfcoder.planner import prioritizer


@pytest.fixture()
def sample_briefs() -> List[ArchitectBrief]:
    high = ArchitectBrief(
        id="brief-high",
        component="core",
        title="Stabilise core",
        summary="High risk",
        priority=5.0,
        rationale=[],
        acceptance_criteria=[],
        signals={
            "hotspot": {"risk_score": 8.5, "test_failures": 1, "apply_failures": 2, "recent_fix_commits": 4},
            "coverage": {"missing_lines": 30, "files": ["core/module.py"]},
            "smells": {"count": 3},
        },
    )
    low = ArchitectBrief(
        id="brief-low",
        component="app",
        title="Clean app",
        summary="Low risk",
        priority=2.0,
        rationale=[],
        acceptance_criteria=[],
        signals={
            "hotspot": {"risk_score": 2.0, "test_failures": 0, "apply_failures": 0},
            "coverage": {"missing_lines": 5, "files": ["app/ui.py"]},
            "smells": {"count": 0},
        },
    )
    return [high, low]


def test_prioritize_briefs_applies_policy(sample_briefs, monkeypatch):
    # Force policy lookup to a known value to avoid relying on config state
    monkeypatch.setattr(prioritizer, "_get_policy", lambda default="balanced": "safe")
    prioritized = prioritizer.prioritize_briefs(sample_briefs)
    decisions = {item.brief.id: item.decision for item in prioritized}
    assert decisions["brief-high"] == "block"
    assert decisions["brief-low"] == "auto"


def test_prioritize_briefs_respects_cost_budget(sample_briefs, monkeypatch):
    monkeypatch.setattr(prioritizer, "_get_policy", lambda default="balanced": "balanced")
    costly = ArchitectBrief(
        id="brief-costly",
        component="ops",
        title="Large refactor",
        summary="Big coverage gap",
        priority=1.0,
        rationale=[],
        acceptance_criteria=[],
        signals={
            "hotspot": {"risk_score": 1.5, "test_failures": 0, "apply_failures": 0},
            "coverage": {"missing_lines": 120, "files": ["ops/telemetry.py"]},
            "smells": {"count": 0},
        },
    )
    monkeypatch.setenv("NERION_ARCHITECT_COST_BUDGET", "200")
    prioritized = prioritizer.prioritize_briefs([costly])
    assert prioritized[0].decision == "review"
    assert any("budget" in reason.lower() for reason in prioritized[0].reasons)


def test_build_planner_context_selects_matching_component(sample_briefs, monkeypatch):
    monkeypatch.setattr(prioritizer, "generate_architect_briefs", lambda **_kw: sample_briefs)
    ctx = prioritizer.build_planner_context(
        instruction="Fix issues in core module",
        target_file="core/models.py",
    )
    assert ctx is not None
    assert ctx["brief"]["component"] == "core"
    assert ctx["decision"] in {"review", "block", "auto"}
    assert ctx["suggested_targets"]
    assert ctx["brief"]["id"].startswith("brief")
