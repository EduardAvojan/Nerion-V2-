import json
from pathlib import Path
from textwrap import dedent

from selfcoder.cli import main
from selfcoder.planner.planner import plan_edits_from_nl
from selfcoder.plans.schema import validate_plan

def test_plan_dry_run_outputs_valid_json(tmp_path, capsys):
    rc = main([
        "plan",
        "-i", "add module docstring 'Planned from test'",
        "-f", str(tmp_path / "dummy.py"),
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    plan = json.loads(out.splitlines()[0])
    assert isinstance(plan, dict)
    assert "actions" in plan and isinstance(plan["actions"], list)
    first_action = plan["actions"][0]
    action_name = first_action.get("kind") or first_action.get("action")
    assert action_name == "add_module_docstring"
    assert first_action.get("payload", {}).get("doc") == "Planned from test"
    assert plan.get("target_file")

def test_plan_apply_writes_docstring(tmp_path, capsys):
    target = tmp_path / "module_under_test.py"
    target.write_text(dedent("""\
        def ping():
            return "pong"
    """), encoding="utf-8")

    rc = main([
        "plan",
        "-i", "add module docstring 'Applied from planner'",
        "-f", str(target),
        "--apply",
        "--force-apply",
    ])
    assert rc == 0

    new_src = target.read_text(encoding="utf-8")
    assert '"""Applied from planner"""' in new_src or "'''Applied from planner'''" in new_src


def test_plan_apply_respects_policy_gate(tmp_path, monkeypatch):
    target = tmp_path / "module_under_test.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    import selfcoder.cli as cli_module
    monkeypatch.setattr(cli_module, "build_planner_context", lambda *a, **k: None, raising=False)

    rc = main([
        "plan",
        "-i", "add module docstring",
        "-f", str(target),
        "--apply",
    ])
    # With no architect metadata the policy should require manual review.
    assert rc == 2
    assert "docstring" not in target.read_text(encoding="utf-8")


def test_plan_includes_coordination_fields(tmp_path):
    target = tmp_path / "sample.py"
    plan = plan_edits_from_nl("add function foo", file=str(target))
    validated = validate_plan(plan)

    # bundle_id should be a non-empty string
    assert isinstance(validated.bundle_id, str) and len(validated.bundle_id) > 0

    # metadata should be a dict with at least a source key
    assert isinstance(validated.metadata, dict)
    assert "source" in validated.metadata

    # preconditions/postconditions are optional, but when present they must be lists of strings
    if validated.preconditions is not None:
        assert isinstance(validated.preconditions, list)
        assert all(isinstance(x, str) for x in validated.preconditions)
    if validated.postconditions is not None:
        assert isinstance(validated.postconditions, list)
        assert all(isinstance(x, str) for x in validated.postconditions)


# --- New tests for planner clarification prompts
def test_plan_from_text_emits_clarify_on_empty_instruction(tmp_path):
    from selfcoder.planner import planner
    plan = planner.plan_from_text("", target_file=str(tmp_path / "dummy.py"))
    assert "clarify" in plan and any("No valid action" in c for c in plan["clarify"])

def test_plan_from_text_emits_clarify_on_missing_symbol_name(tmp_path):
    from selfcoder.planner import planner
    # Action with insert_function but no name
    plan = {
        "actions": [{"kind": "insert_function", "payload": {"doc": "missing name"}}],
        "target_file": str(tmp_path / "dummy.py"),
        "metadata": {},
    }
    out = planner.plan_from_text("insert function", target_file=str(tmp_path / "dummy.py"))
    # We expect clarify about missing symbol name or target
    assert "clarify" in out
    assert any("Missing symbol name" in c or "No target file" in c for c in out["clarify"])

def test_plan_edits_from_nl_sets_clarify_required_flag(tmp_path):
    from selfcoder.planner.planner import plan_edits_from_nl
    plan = plan_edits_from_nl("", file=str(tmp_path / "dummy.py"))
    if "clarify" in plan:
        assert plan.get("metadata", {}).get("clarify_required") is True


def test_plan_edits_uses_brief_context_for_target(tmp_path):
    from selfcoder.planner.planner import plan_edits_from_nl

    brief_context = {
        "brief": {
            "id": "brief-test",
            "component": "core",
            "title": "Stabilise core",
            "summary": "Fix telemetry hotspots",
        },
        "decision": "auto",
        "policy": "balanced",
        "risk_score": 3.5,
        "effort_score": 1.2,
        "estimated_cost": 120.0,
        "effective_priority": 7.5,
        "reasons": ["Risk score exceeds review threshold"],
        "suggested_targets": ["core/example.py"],
        "alternates": [],
        "gating": {"risk": {"score": 3.5}},
    }

    plan = plan_edits_from_nl("add function foo", brief_context=brief_context)
    assert plan.get("target_file") == "core/example.py"
    meta = plan.get("metadata", {}).get("architect_brief")
    assert meta is not None
    assert meta["id"] == "brief-test"
    assert "core/example.py" in meta.get("suggested_targets", [])
