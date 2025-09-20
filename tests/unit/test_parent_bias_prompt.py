from __future__ import annotations

import json
from types import SimpleNamespace

import app.parent.driver as drv


class _FakeLLM(drv.ParentLLM):
    def complete(self, messages):
        # Return a minimal valid ParentDecision
        return json.dumps({
            "intent": "respond",
            "plan": [{"action": "respond", "tool": None, "args": {}, "summary": "done"}],
            "final_response": "ok",
            "confidence": 0.5,
            "requires_network": False,
            "notes": None,
        })


def test_parent_bias_includes_tool_success_rates(monkeypatch):
    # Stub learned prefs with a couple of tools
    monkeypatch.setattr(drv, "_load_prefs", lambda: {"tool_success_rate": {"read_file": 0.9, "web_search": 0.6}}, raising=False)

    captured = {}

    def _fake_build_master_prompt(user_query, tools, context_snippet=None, extra_policies=None):
        captured["policies"] = extra_policies or ""
        return {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]}

    monkeypatch.setattr(drv, "build_master_prompt", _fake_build_master_prompt, raising=False)

    tman = drv.ToolsManifest(tools=[])
    p = drv.ParentDriver(llm=_FakeLLM(), tools=tman)
    _ = p.plan_and_route(user_query="test")

    pol = captured.get("policies", "")
    assert "Success rates" in pol or "success rate" in pol
    assert "read_file" in pol

