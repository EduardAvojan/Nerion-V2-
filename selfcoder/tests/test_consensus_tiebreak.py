

from __future__ import annotations

from selfcoder.analysis.consensus.aggregate import aggregate_consensus


def test_consensus_prefers_external_with_recency():
    evidences = [
        {"summary": "Onsite says Model A is highlighted.", "confidence": 0.80, "source": "onsite", "url": "https://brand.example/a"},
        {"summary": "Independent review: Model B leads this month.", "confidence": 0.79, "source": "external", "url": "https://reviews.example/b", "date_hint": "2025-07-15", "name": "Model B"},
    ]
    out = aggregate_consensus(evidences)
    win = out.get("winner", {})
    assert win.get("source") == "external"
    assert win.get("name") == "Model B"


def test_consensus_tie_break_favors_external_then_recency():
    # Equal confidence, expect external win; if equal and no external, recency wins
    evidences = [
        {"summary": "Model C highlighted.", "confidence": 0.80, "source": "onsite", "url": "https://brand.example/c"},
        {"summary": "Model D review.", "confidence": 0.80, "source": "external", "url": "https://reviews.example/d", "date_hint": "2025-07-20", "name": "Model D"},
    ]
    out = aggregate_consensus(evidences)
    win = out.get("winner", {})
    assert win.get("source") == "external"
    assert win.get("name") == "Model D"