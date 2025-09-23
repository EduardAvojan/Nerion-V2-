from __future__ import annotations

import os
import json
from pathlib import Path
import pytest

from selfcoder.planner.utils import sanitize_plan, repo_fingerprint, load_plan_cache, save_plan_cache


def test_sanitize_drops_disallowed_and_normalizes(tmp_path: Path):
    raw = {
        "actions": [
            {"kind": "create_file", "payload": {"path": "a.py", "content": "print(1)"}},
            {"kind": "unknown_action", "payload": {}},
        ],
        "metadata": {"foo": "bar"},
    }
    clean = sanitize_plan(raw)
    assert isinstance(clean, dict)
    acts = clean.get("actions")
    assert isinstance(acts, list) and len(acts) == 1
    a0 = acts[0]
    assert a0.get("kind") == "create_file"
    assert "payload" in a0 and a0["payload"].get("path") == "a.py"


def test_repo_fingerprint_changes_with_file(tmp_path: Path):
    (tmp_path / "x.py").write_text("x=1\n", encoding="utf-8")
    f1 = repo_fingerprint(tmp_path)
    (tmp_path / "x.py").write_text("x=2\n", encoding="utf-8")
    f2 = repo_fingerprint(tmp_path)
    assert f1 != f2


def test_cache_roundtrip(tmp_path: Path):
    p = tmp_path / "cache.json"
    data = {"k": {"actions": []}}
    save_plan_cache(p, data)
    got = load_plan_cache(p)
    assert got == data


@pytest.mark.skipif(bool(os.environ.get("NERION_V2_CODE_PROVIDER")), reason="LLM may be available")
def test_llm_strict_raises_when_unavailable():
    os.environ["NERION_LLM_STRICT"] = "1"
    from selfcoder.planner.llm_planner import plan_with_llm
    with pytest.raises(RuntimeError):
        plan_with_llm("add docstring", None)
    os.environ.pop("NERION_LLM_STRICT", None)


def test_plan_with_llm_includes_brief_context(monkeypatch):
    from selfcoder.planner import llm_planner

    captured = {}

    class DummyCoder:
        def __init__(self, role: str | None = None):
            captured["role"] = role

        def complete_json(self, prompt: str, system: str) -> str:
            captured["prompt"] = prompt
            return json.dumps(
                {
                    "actions": [
                        {"kind": "create_file", "payload": {"path": "core/example.py"}}
                    ],
                    "target_file": "core/example.py",
                }
            )

    monkeypatch.setattr(llm_planner, "Coder", DummyCoder)

    context = {
        "brief": {
            "id": "brief-llm",
            "component": "core",
            "title": "Core upgrade",
            "summary": "Improve reliability",
            "rationale": ["Risk rising"],
            "acceptance_criteria": ["Telemetry quiet"],
        },
        "decision": "review",
        "policy": "safe",
        "risk_score": 9.0,
        "effort_score": 2.5,
        "estimated_cost": 400.0,
        "effective_priority": 12.0,
        "reasons": ["Risk score 9.0 exceeds review threshold"],
        "suggested_targets": ["core/example.py"],
        "alternates": [],
        "gating": {"risk": {"score": 9.0}},
    }

    plan = llm_planner.plan_with_llm("add function foo", None, brief_context=context)

    assert "Architect brief context" in captured.get("prompt", "")
    assert "core" in captured.get("prompt", "")
    meta = plan.get("metadata", {}).get("architect_brief")
    assert meta and meta["id"] == "brief-llm"
    assert plan.get("target_file") == "core/example.py"
