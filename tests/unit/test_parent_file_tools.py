import os, tempfile
from app.chat.parent_exec import build_executor
from app.parent.schemas import ParentDecision, Step


def _no_net(task_type, url):
    return True


def _dummy_heard():
    return "dummy"


def _parse_slots(_: str):
    return {}


def test_read_file_tool_smoke():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sample.txt")
        with open(p, 'w', encoding='utf-8') as f:
            f.write("hello world\nthis is a test\n")
        ex = build_executor(ensure_network_for=_no_net, get_heard=_dummy_heard, parse_task_slots=_parse_slots)
        dec = ParentDecision(intent="files.read",
                             plan=[Step(action="tool_call", tool="read_file", args={"path": p})],
                             final_response=None,
                             confidence=0.4,
                             requires_network=False)
        out = ex.execute(dec, user_query="read file")
        assert out["success"] is True
        steps = out["action_taken"]["steps"]
        assert steps and steps[0]["tool"] == "read_file"


def test_summarize_file_tool_fallback(monkeypatch):
    # Force heuristic fallback (no LLM) and balanced policy to avoid env interference
    monkeypatch.setenv("NERION_SUMMARY_LLM", "0")
    monkeypatch.setenv("NERION_POLICY", "balanced")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sample.txt")
        with open(p, 'w', encoding='utf-8') as f:
            f.write("line1\nline2\nline3\nline4\nline5\nline6\n")
        ex = build_executor(ensure_network_for=_no_net, get_heard=_dummy_heard, parse_task_slots=_parse_slots)
        dec = ParentDecision(intent="files.sum",
                             plan=[Step(action="tool_call", tool="summarize_file", args={"path": p})],
                             final_response=None,
                             confidence=0.4,
                             requires_network=False)
        out = ex.execute(dec, user_query="summarize file")
        assert out["success"] is True
        steps = out["action_taken"]["steps"]
        assert steps and steps[0]["tool"] == "summarize_file"
