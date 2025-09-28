from __future__ import annotations

from typing import Any, Dict

from app.parent.executor import ParentExecutor
from app.parent.schemas import ParentDecision, Step


def test_parent_executor_read_url_arg_validation():
    # Runner stub (won't be called due to invalid args)
    def _read_url(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    ex = ParentExecutor(tool_runners={"read_url": _read_url}, ensure_network=None)
    dec = ParentDecision(
        intent="web.site_query",
        plan=[Step(action="tool_call", tool="read_url", args={"url": "not-a-valid-url", "timeout": 5})],
        final_response=None,
        confidence=0.5,
        requires_network=False,
        notes=None,
    )
    out = ex.execute(decision=dec, user_query="read bad url")
    assert out.get("success") is False
    assert "invalid_args" in (out.get("error") or "")


def test_parent_executor_success_and_metrics():
    calls = []

    def _read_url(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate quick tool success
        return {"ok": True, "url": kwargs.get("url")}

    def _metric(tool: str, ok: bool, dur: float, err: Optional[str]):
        calls.append({"tool": tool, "ok": ok, "dur": dur, "err": err})

    ex = ParentExecutor(
        tool_runners={"read_url": _read_url},
        ensure_network=None,
        allowed_tools=["read_url"],
        metrics_hook=_metric,
    )
    dec = ParentDecision(
        intent="web.site_query",
        plan=[Step(action="tool_call", tool="read_url", args={"url": "https://example.com", "timeout": 3})],
        final_response=None,
        confidence=0.7,
        requires_network=False,
        notes=None,
    )
    out = ex.execute(decision=dec, user_query="read good url")
    assert out.get("success") is True
    # Metrics called once with expected values
    assert len(calls) == 1
    m = calls[0]
    assert m["tool"] == "read_url"
    assert m["ok"] is True
    assert isinstance(m["dur"], float) and m["dur"] >= 0.0
    assert m["err"] in (None, "")
