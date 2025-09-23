import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.voice


_RELAXED_POLICY = Path(__file__).parent / "fixtures" / "policy_relaxed.yaml"


@pytest.fixture(autouse=True)
def _relaxed_policy(monkeypatch):
    monkeypatch.setenv("NERION_POLICY_FILE", str(_RELAXED_POLICY))
    monkeypatch.setenv("NERION_GOVERNOR_MIN_INTERVAL_MINUTES", "0")
    monkeypatch.setenv("NERION_GOVERNOR_MAX_RUNS_PER_HOUR", "0")
    monkeypatch.setenv("NERION_GOVERNOR_MAX_RUNS_PER_DAY", "0")

@pytest.mark.skipif(
    not hasattr(importlib.util.find_spec("app.nerion_chat"), "loader"),
    reason="app.nerion_chat module not available"
)
def test_voice_pipeline_applies_docstring(tmp_path, monkeypatch):
    import importlib
    nc = importlib.import_module("app.nerion_chat")

    if not hasattr(nc, "run_self_coding_pipeline"):
        pytest.skip("run_self_coding_pipeline not available in app.nerion_chat")

    # Create a target file OUTSIDE the repo (tmp_path is fine)
    target = tmp_path / "voice_target.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")

    # Force planner to use heuristic stub and avoid real provider calls
    monkeypatch.setenv("NERION_USE_CODER_LLM", "0")

    import selfcoder.planner.prioritizer as prioritizer_mod

    monkeypatch.setattr(prioritizer_mod, "build_planner_context", lambda *a, **k: None)

    import selfcoder.planner.planner as planner_mod

    def _fake_plan(instruction, _target, *, brief_context=None):
        return {
            "actions": [{"kind": "noop"}],
            "target_file": str(target),
            "metadata": {
                "architect_brief": {
                    "decision": "auto",
                    "policy": "fast",
                }
            },
        }

    monkeypatch.setattr(planner_mod, "plan_edits_from_nl", _fake_plan)

    import selfcoder.planner.apply_policy as apply_policy_mod

    monkeypatch.setattr(
        apply_policy_mod,
        "evaluate_apply_policy",
        lambda plan, **kwargs: apply_policy_mod.ApplyPolicyDecision(decision="auto", policy="fast"),
    )
    monkeypatch.setattr(apply_policy_mod, "apply_allowed", lambda decision, **kwargs: True)

    import app.chat.self_coding as sc_mod

    class _GovernorOK:
        def __init__(self):
            self.allowed = True
            self.code = "ok"
            self.override_used = False
            self.reasons = []
            self.next_allowed_local = None

        def is_blocked(self):
            return False

    monkeypatch.setattr(sc_mod, "governor_evaluate", lambda *a, **k: _GovernorOK())
    monkeypatch.setattr(sc_mod, "governor_note_execution", lambda *a, **k: None)

    import selfcoder.orchestrator as orchestrator_mod

    def _fake_apply(plan, dry_run=False, preview=False, healers=None):
        p = Path(plan.get("target_file"))
        existing = p.read_text(encoding="utf-8")
        p.write_text('"""Voice Test"""\n' + existing, encoding="utf-8")
        return [p]

    monkeypatch.setattr(orchestrator_mod, "apply_plan", _fake_apply)

    # Stub out I/O: no speaking, and "hear" the path to our file
    nc.speak = lambda *a, **k: None
    nc.listen_once = lambda **k: str(target)

    # Run the pipeline with a simple instruction
    ok = nc.run_self_coding_pipeline("add module docstring 'Voice Test'")
    assert ok is True

    # Verify the file received the docstring
    new_src = target.read_text(encoding="utf-8")
    assert '"""Voice Test"""' in new_src or "'''Voice Test'''" in new_src
