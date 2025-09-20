from __future__ import annotations

import json

from app.parent.driver import ParentLLM, ParentDriver
from app.parent.schemas import ParentDecision, Step


class _FakeLLM(ParentLLM):
    def complete(self, messages):
        return json.dumps({
            "intent": "tools",
            "plan": [{"action": "tool_call", "tool": "t1", "args": {}, "summary": "run t1"}],
            "final_response": None,
            "confidence": 0.5,
            "requires_network": False,
            "notes": None,
        })


def test_policy_balanced_retries(monkeypatch):
    calls = {"n": 0}
    def runner(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "ok"

    driver = ParentDriver(llm=_FakeLLM(), tools=type('T', (), {'tools': []})())
    from app.parent.executor import ParentExecutor
    # Ensure balanced policy so a single retry is attempted (env may override)
    monkeypatch.setenv("NERION_POLICY", "balanced")
    ex = ParentExecutor(tool_runners={"t1": runner})
    dec = json.loads(_FakeLLM().complete([]))
    out = ex.execute(ParentDecision(**dec), user_query="go")
    assert out["success"] is True
    assert calls["n"] == 2  # default retry once in balanced policy


def test_policy_safe_no_retry(monkeypatch):
    calls = {"n": 0}
    def runner(**kw):
        calls["n"] += 1
        raise RuntimeError("boom")

    # Set safe policy
    monkeypatch.setenv("NERION_POLICY", "safe")
    from app.parent.executor import ParentExecutor
    ex = ParentExecutor(tool_runners={"t1": runner})
    dec = json.loads(_FakeLLM().complete([]))
    out = ex.execute(ParentDecision(**dec), user_query="go")
    assert out["success"] is False
    assert calls["n"] == 1  # no retry under safe policy
