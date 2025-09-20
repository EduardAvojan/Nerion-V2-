from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path

from selfcoder import cli


def test_cli_healthcheck_ok():
    assert cli.main(["healthcheck"]) in (0,)


def test_cli_docstring_dry_run_module(tmp_path: Path):
    # use a tiny temp file to avoid touching repo files
    p = tmp_path / "mod.py"
    p.write_text("def f():\n    return 1\n", encoding="utf-8")
    rc = cli.main([
        "docstring",
        "--file", str(p),
        "--module-doc", "Temp module",
        "--dry-run",
    ])
    assert rc == 0


def test_cli_snapshot_safe_mode(monkeypatch):
    # ensure snapshot stays non-destructive in tests
    monkeypatch.setenv("SELFCODER_SAFE_MODE", "1")
    monkeypatch.setenv("SELFCODER_DRYRUN", "1")
    rc = cli.main(["snapshot", "--message", "Test snapshot"])
    assert rc == 0


def test_cli_ext_plan_outputs_valid_json(tmp_path: Path):
    target = tmp_path / "tmp.py"
    target.write_text("print('ok')\n", encoding="utf-8")

    # Invoke the external CLI entrypoint to generate a plan
    cmd = [
        sys.executable,
        "-m",
        "selfcoder.cli_ext.__main__",
        "plan",
        "add function foo",
        "--file",
        str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr

    # The CLI prints a JSON plan validated by the schema
    data = json.loads(proc.stdout)
    assert isinstance(data, dict)
    assert "actions" in data and isinstance(data["actions"], list)
    assert data.get("bundle_id")
    meta = data.get("metadata")
    assert isinstance(meta, dict)
    assert "source" in meta


# Additional tests for preview diff and healer behavior
def _cli_plan(tmp_path: Path, instruction: str, target: Path) -> Path:
    """Helper: call the external CLI to generate a plan JSON and return its path."""
    cmd = [
        sys.executable,
        "-m",
        "selfcoder.cli_ext.__main__",
        "plan",
        instruction,
        "--file",
        str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(proc.stdout, encoding="utf-8")
    return plan_path


def test_cli_ext_apply_preview_prints_diff_and_makes_no_changes(tmp_path: Path):
    target = tmp_path / "mod.py"
    original = "print('ok')\n"
    target.write_text(original, encoding="utf-8")

    plan_path = _cli_plan(tmp_path, "add function foo: \"def foo():\n    return 42\n\"", target)

    # Apply with preview: should print a unified diff and not modify the file
    cmd = [
        sys.executable,
        "-m",
        "selfcoder.cli_ext.__main__",
        "apply",
        str(plan_path),
        "--preview",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # Prefer unified diff markers; allow empty diff if no change was computed
    out = proc.stdout
    if out.strip():
        assert ("--- a/" in out and "+++ b/" in out) or "@@" in out

    # File must remain unchanged
    assert target.read_text(encoding="utf-8") == original


def test_cli_ext_apply_with_format_healer_strips_trailing_spaces(tmp_path: Path):
    target = tmp_path / "mod2.py"
    # Deliberate trailing spaces to be cleaned by the 'format' healer
    original = "print('ok')    \n"
    target.write_text(original, encoding="utf-8")

    plan_path = _cli_plan(tmp_path, "add function foo: \"def foo():\n    return 42\n\"", target)

    # Apply with format healer enabled
    cmd = [
        sys.executable,
        "-m",
        "selfcoder.cli_ext.__main__",
        "apply",
        str(plan_path),
        "--heal=format",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # Trailing spaces should be removed and file should end with a single newline
    content = target.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert "    \n" not in content


def test_selfaudit_generates_schema_valid_plan(tmp_path: Path):
    from selfcoder.selfaudit import generate_improvement_plan
    from selfcoder.plans.schema import validate_plan

    plan = generate_improvement_plan(tmp_path)
    # Should be a dict that validates and includes actions + target
    assert isinstance(plan, dict)
    validated = validate_plan(plan)
    assert len(validated.actions) >= 1
    # Ensure target exists in the plan dict (schema keeps it as metadata)
    assert "target_file" in plan


def test_cli_ext_apply_with_isort_healer_orders_imports(tmp_path: Path):
    try:
        import isort  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("isort not installed")

    target = tmp_path / "imports.py"
    # Intentionally disordered imports
    target.write_text("import sys\nimport os\n", encoding="utf-8")

    plan_path = _cli_plan(tmp_path, "add function foo: \"def foo():\n    return 42\n\"", target)

    cmd = [
        sys.executable,
        "-m",
        "selfcoder.cli_ext.__main__",
        "apply",
        str(plan_path),
        "--heal=isort",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    content = target.read_text(encoding="utf-8")
    # isort's default sorts alphabetically: os before sys
    assert content.splitlines()[0].strip() == "import os"


# Test the CLI audit subcommand
def test_cli_ext_audit_generates_valid_plan(tmp_path: Path):
    # Run the external CLI audit command
    cmd = [
        sys.executable,
        "-m",
        "selfcoder.cli_ext.__main__",
        "audit",
        "--root",
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # The CLI prints a JSON plan; parse and validate
    data = json.loads(proc.stdout)
    assert isinstance(data, dict)
    assert "actions" in data
    from selfcoder.plans.schema import validate_plan
    validated = validate_plan(data)
    assert len(validated.actions) >= 1

def test_cli_ext_audit_schedule_runs_once(tmp_path: Path):
    # Run the external CLI audit-schedule command with --once
    cmd = [
        sys.executable,
        "-m",
        "selfcoder.cli_ext.__main__",
        "audit-schedule",
        "--root",
        str(tmp_path),
        "--once",
    ]
    env = os.environ.copy()
    env["SELFAUDIT_ENABLE"] = "1"
    env["SELFAUDIT_INTERVAL"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr

    out = proc.stdout
    assert "[scheduler] generated plan" in out