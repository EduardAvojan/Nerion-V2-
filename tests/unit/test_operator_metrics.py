from __future__ import annotations

from ops.telemetry import operator


def test_compute_apply_metrics_counts():
    events = [
        {"metadata": {"outcome": True, "rolled_back": False, "simulate": False}},
        {"metadata": {"outcome": False, "rolled_back": True, "simulate": False}},
        {"metadata": {"outcome": True, "rolled_back": False, "simulate": True}},
    ]
    metrics = operator._compute_apply_metrics(events)
    assert metrics["total"] == 3
    assert metrics["success"] == 2
    assert metrics["rolled_back"] == 1
    assert metrics["simulated"] == 1
    assert 0.0 < metrics["rate"] < 1.0


def test_summarize_snapshot_includes_apply_metrics():
    snapshot = {
        "window": {"hours": 6},
        "counts_total": 12,
        "prompt_completion_ratio": {"prompts": 5, "completions": 4},
        "providers": [],
        "anomalies": [],
        "apply_metrics": {"total": 3, "success": 2, "rolled_back": 1, "simulated": 0, "rate": 0.666666},
        "policy_gates": {"auto": 2, "review": 1},
        "governor_decisions": {},
        "provider_cost_total": 1.2345,
    }
    summary = operator.summarize_snapshot(snapshot)
    labels = {item["label"] for item in summary["metrics"]}
    assert "Apply success" in labels
    assert "Rollbacks" in labels
    assert "Cost window" in labels
    assert summary["subtitle"].startswith("2/3 applies")
