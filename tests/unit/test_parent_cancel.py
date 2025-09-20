from app.parent.executor import ParentExecutor
from app.parent.schemas import ParentDecision, Step


def test_cancel_check_aborts_before_steps():
    called = {"t": 0}
    def t(**kw):
        called["t"] += 1
        return {"ok": True}

    ex = ParentExecutor(tool_runners={"t": t}, cancel_check=lambda: True)
    dec = ParentDecision(intent="multi",
                         plan=[Step(action="tool_call", tool="t", args={})],
                         final_response=None,
                         confidence=0.5,
                         requires_network=False)
    out = ex.execute(dec, user_query="cancel please")
    assert out["success"] is False
    assert "cancelled" in (out.get("error") or "")
    assert called["t"] == 0

