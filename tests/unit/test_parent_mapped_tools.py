from app.chat.parent_exec import build_executor
from app.parent.schemas import ParentDecision, Step


def _no_net(task_type, url):
    return True


def _dummy_heard():
    return "dummy"


def _parse_slots(_: str):
    return {}


def test_executor_list_plugins_smoke():
    ex = build_executor(ensure_network_for=_no_net, get_heard=_dummy_heard, parse_task_slots=_parse_slots)
    dec = ParentDecision(intent="tools.list",
                         plan=[Step(action="tool_call", tool="list_plugins", args={})],
                         final_response=None,
                         confidence=0.5,
                         requires_network=False)
    out = ex.execute(dec, user_query="list plugins")
    assert out["success"] is True
    steps = out["action_taken"]["steps"]
    assert steps and steps[0]["tool"] == "list_plugins"


def test_executor_run_pytest_smoke_with_stub(monkeypatch):
    # Stub safe_run to avoid running pytest in test
    import app.chat.parent_exec as pe
    class R:
        returncode = 0
        stdout = b"collected 0 items\n\nok\n"
        stderr = b""
    monkeypatch.setattr(pe, "_safe_run", lambda *a, **k: R())
    ex = pe.build_executor(ensure_network_for=_no_net, get_heard=_dummy_heard, parse_task_slots=_parse_slots)
    dec = ParentDecision(intent="tools.smoke",
                         plan=[Step(action="tool_call", tool="run_pytest_smoke", args={})],
                         final_response=None,
                         confidence=0.5,
                         requires_network=False)
    out = ex.execute(dec, user_query="run smoke tests")
    assert out["success"] is True
    steps = out["action_taken"]["steps"]
    assert steps and steps[0]["tool"] == "run_pytest_smoke"

