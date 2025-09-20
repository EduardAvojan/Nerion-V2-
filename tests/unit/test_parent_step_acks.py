from app.parent.executor import ParentExecutor
from app.parent.schemas import ParentDecision, Step


def test_parent_executor_prints_step_acks(capsys):
    ex = ParentExecutor(tool_runners={"t": lambda **kw: {"ok": True}})
    dec = ParentDecision(intent="multi",
                         plan=[
                             Step(action="tool_call", tool="t", args={}, summary="first"),
                             Step(action="tool_call", tool="t", args={}, summary="second"),
                         ],
                         final_response=None,
                         confidence=0.5,
                         requires_network=False)
    out = ex.execute(dec, user_query="do things")
    assert out["success"] is True
    std = capsys.readouterr().out
    assert "Step 1/2" in std and "Step 2/2" in std

