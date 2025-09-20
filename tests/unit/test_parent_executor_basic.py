import types
import pytest

from app.parent.executor import ParentExecutor
from app.parent.schemas import ParentDecision, Step


def test_executor_runs_healthcheck_tool():
    def run_healthcheck(**kw):
        return "Healthcheck: OK."

    ex = ParentExecutor(tool_runners={"run_healthcheck": run_healthcheck})
    dec = ParentDecision(intent="system.health",
                         plan=[Step(action="tool_call", tool="run_healthcheck", args={})],
                         final_response=None,
                         confidence=0.9,
                         requires_network=False)
    out = ex.execute(dec, user_query="run healthcheck")
    assert out["success"] is True
    assert out["action_taken"]["steps"][0]["tool"] == "run_healthcheck"


def test_executor_continue_on_error_and_timeout(monkeypatch):
    calls = {"a": 0, "b": 0}

    def slow_fail(**kw):
        calls["a"] += 1
        raise RuntimeError("boom")

    def ok(**kw):
        calls["b"] += 1
        return {"ok": True}

    ex = ParentExecutor(tool_runners={"t1": slow_fail, "t2": ok})
    dec = ParentDecision(intent="multi",
                         plan=[
                             Step(action="tool_call", tool="t1", args={}, continue_on_error=True, timeout_s=0.01),
                             Step(action="tool_call", tool="t2", args={}),
                         ],
                         final_response=None,
                         confidence=0.5,
                         requires_network=False)
    out = ex.execute(dec, user_query="do things")
    assert out["success"] is True
    # both attempted
    assert calls["a"] == 1 and calls["b"] == 1

