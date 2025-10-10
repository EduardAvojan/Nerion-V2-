import json
import subprocess
import sys
from pathlib import Path
import os
import shutil


_RELAXED_POLICY = Path(__file__).parent / "fixtures" / "policy_relaxed.yaml"

def find_project_root() -> Path:
    """Finds the project root directory by looking for a known file like pyproject.toml or .git"""
    start_dir = Path(__file__).parent
    for parent in [start_dir, *start_dir.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise FileNotFoundError("Could not find the project root.")

PROJECT_ROOT = find_project_root()

def setup_test_project(root_path: Path):
    """
    Helper function to create a minimal, runnable project structure.
    """
    project_dir = root_path / "project"
    project_dir.mkdir()
    
    selfcoder_dir = project_dir / "selfcoder"
    
    for d_name in ["", "planner", "actions", "vcs", "tests"]:
        (selfcoder_dir / d_name).mkdir(parents=True, exist_ok=True)
    
    for d_name in ["", "planner", "actions", "vcs", "tests"]:
        (selfcoder_dir / d_name / "__init__.py").touch()
    
    (project_dir / "app").mkdir(exist_ok=True)
    (project_dir / "app" / "__init__.py").touch()
    
    # Copy real files from the actual project source
    # cli is now a package, not a single file
    shutil.copytree(PROJECT_ROOT / "selfcoder/cli", selfcoder_dir / "cli", dirs_exist_ok=True)
    shutil.copy(PROJECT_ROOT / "selfcoder/simulation.py", selfcoder_dir / "simulation.py")
    shutil.copy(PROJECT_ROOT / "selfcoder/actions/crossfile.py", selfcoder_dir / "actions" / "crossfile.py")

    # Create dummy modules that are imported by the CLI
    (selfcoder_dir / "healthcheck.py").write_text("def run_all(): return (True, 'OK')")
    (selfcoder_dir / "orchestrator.py").write_text("def run_actions_on_file(*a, **kw): return True")
    (selfcoder_dir / "planner" / "planner.py").write_text("def plan_edits_from_nl(*a, **kw): return {'actions': [], 'target_file': 'app.py'}")
    (selfcoder_dir / "planner" / "prioritizer.py").write_text("def build_planner_context(*a, **kw):\n    return None\n")
    (selfcoder_dir / "planner" / "apply_policy.py").write_text(
        "class ApplyPolicyDecision:\n"
        "    def __init__(self, decision='auto', reasons=None, policy='fast'):\n"
        "        self.decision = decision\n"
        "        self.reasons = list(reasons or [])\n"
        "        self.policy = policy\n"
        "    def is_blocked(self):\n"
        "        return self.decision == 'block'\n"
        "    def requires_manual_review(self):\n"
        "        return self.decision == 'review'\n"
        "\n"
        "def evaluate_apply_policy(plan, policy=None, default_decision='auto'):\n"
        "    meta = plan.get('metadata') if isinstance(plan, dict) else {}\n"
        "    brief = meta.get('architect_brief') if isinstance(meta, dict) else {}\n"
        "    decision = brief.get('decision') if isinstance(brief, dict) else None\n"
        "    if not decision:\n"
        "        decision = default_decision or 'auto'\n"
        "    use_policy = policy or (brief.get('policy') if isinstance(brief, dict) else None) or 'fast'\n"
        "    reasons = brief.get('reasons') if isinstance(brief, dict) else None\n"
        "    return ApplyPolicyDecision(decision=decision, reasons=reasons, policy=use_policy)\n"
        "\n"
        "def apply_allowed(decision, allow_review=False, force=False):\n"
        "    if force:\n"
        "        return True\n"
        "    if decision.is_blocked():\n"
        "        return False\n"
        "    if decision.requires_manual_review():\n"
        "        return allow_review\n"
        "    return True\n"
    )
    (selfcoder_dir / "planner" / "utils.py").write_text(
        "def attach_brief_metadata(plan, brief_context):\n    return plan\n"
    )
    (selfcoder_dir / "vcs" / "git_ops.py").write_text("def snapshot(*a, **kw): return 'ts'\ndef restore_snapshot(*a, **kw): pass")

    return project_dir

def run_cli_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Helper to run a CLI command with the correct environment."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("NERION_POLICY_FILE", str(_RELAXED_POLICY))
    env.setdefault("NERION_GOVERNOR_MIN_INTERVAL_MINUTES", "0")
    env.setdefault("NERION_GOVERNOR_MAX_RUNS_PER_HOUR", "0")
    env.setdefault("NERION_GOVERNOR_MAX_RUNS_PER_DAY", "0")
    return subprocess.run(command, capture_output=True, text=True, cwd=cwd, env=env)


def test_simulate_shows_diff_and_does_not_change_original(tmp_path: Path):
    """
    Tests that --simulate runs, shows a diff, and leaves the original file untouched.
    """
    project_dir = setup_test_project(tmp_path)
    file_to_change = project_dir / "app.py"
    original_content = "import old.mod\n"
    file_to_change.write_text(original_content)

    result = run_cli_command([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod", "--new", "new.mod", "--apply", "--simulate", str(file_to_change)
    ], cwd=project_dir)

    assert result.returncode == 0
    assert "--- Simulation Report ---" in result.stdout
    assert file_to_change.read_text() == original_content

def test_simulation_fails_if_tests_fail_in_shadow(tmp_path: Path):
    """
    Tests that the simulation returns a non-zero exit code if tests fail.
    """
    project_dir = setup_test_project(tmp_path)
    app_file = project_dir / "app.py"
    app_file.write_text("print('hello')")
    
    failing_test = project_dir / "selfcoder/tests/test_will_fail.py"
    failing_test.write_text("def test_failing():\n    assert 0\n")

    result = run_cli_command([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod", "--new", "new.mod", "--apply", "--simulate", str(app_file)
    ], cwd=project_dir)

    assert result.returncode != 0
    assert "[simulate] Pytest exit code: 1" in result.stdout

def test_simulation_fails_if_healthcheck_fails_in_shadow(tmp_path: Path):
    """
    Tests that the simulation returns a non-zero exit code if healthcheck fails.
    """
    project_dir = setup_test_project(tmp_path)
    app_file = project_dir / "app.py"
    app_file.write_text("print('hello')")

    (project_dir / "selfcoder" / "healthcheck.py").write_text("def run_all(): return (False, 'Mock Failure')")

    result = run_cli_command([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod", "--new", "new.mod", "--apply", "--simulate", str(app_file)
    ], cwd=project_dir)

    assert result.returncode != 0
    assert "[simulate] Healthcheck exit code: 1" in result.stdout


def test_simulation_optional_check_success(tmp_path: Path, monkeypatch):
    project_dir = setup_test_project(tmp_path)
    file_to_change = project_dir / "app.py"
    file_to_change.write_text("import old.mod\n")

    lint_cmd = f"{sys.executable} -c \"print('lint ok')\""
    monkeypatch.setenv("NERION_SIM_LINT_CMD", lint_cmd)
    monkeypatch.setenv("NERION_SIM_TYPE_CMD", "skip")
    monkeypatch.setenv("NERION_SIM_UI_CMD", "skip")
    monkeypatch.setenv("NERION_SIM_REG_CMD", "skip")

    result = run_cli_command([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod", "--new", "new.mod",
        "--apply", "--simulate", "--simulate-json",
        "--skip-pytest", "--skip-healthcheck",
        str(file_to_change)
    ], cwd=project_dir)

    assert result.returncode == 0
    lines = result.stdout.splitlines()
    start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('{'))
    json_blob = "\n".join(lines[start_idx:])
    payload = json.loads(json_blob)
    lint_report = payload["checks"]["lint"]
    assert lint_report.get("skipped") is False
    assert lint_report.get("rc") == 0
    assert "lint ok" in lint_report.get("stdout", "")


def test_simulation_optional_check_failure(tmp_path: Path, monkeypatch):
    project_dir = setup_test_project(tmp_path)
    file_to_change = project_dir / "app.py"
    file_to_change.write_text("import old.mod\n")

    failing_cmd = f"{sys.executable} -c \"import sys; sys.exit(2)\""
    monkeypatch.setenv("NERION_SIM_LINT_CMD", failing_cmd)
    monkeypatch.setenv("NERION_SIM_TYPE_CMD", "skip")
    monkeypatch.setenv("NERION_SIM_UI_CMD", "skip")
    monkeypatch.setenv("NERION_SIM_REG_CMD", "skip")

    result = run_cli_command([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod", "--new", "new.mod",
        "--apply", "--simulate",
        "--skip-pytest", "--skip-healthcheck",
        str(file_to_change)
    ], cwd=project_dir)

    assert result.returncode != 0
    assert "[simulate] Lint exit code: 2" in result.stdout


def test_simulation_json_and_skip_flags(tmp_path: Path):
    """
    Verify that `--simulate-json` produces a structured report and that
    `--skip-pytest/--skip-healthcheck` avoid running them.
    """
    project_dir = setup_test_project(tmp_path)
    file_to_change = project_dir / "app.py"
    file_to_change.write_text("import old.mod\n")

    result = run_cli_command([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod", "--new", "new.mod",
        "--apply", "--simulate", "--simulate-json",
        "--skip-pytest", "--skip-healthcheck",
        str(file_to_change)
    ], cwd=project_dir)

    # Should succeed because command exit is 0 and both checks are skipped
    assert result.returncode == 0

    # The CLI prints a human report followed by a JSON blob; parse the last JSON object
    # Extract the last line that looks like a JSON closing brace block
    stdout = result.stdout
    lines = stdout.splitlines()
    start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('{'))
    json_blob = "\n".join(lines[start_idx:])

    import json as _json
    payload = _json.loads(json_blob)

    # Basic shape checks
    assert isinstance(payload, dict)
    assert "shadow_dir" in payload and isinstance(payload["shadow_dir"], str)
    assert payload.get("cmd_exit") == 0
    assert "changed" in payload and isinstance(payload["changed"], bool)
    assert "changed_files" in payload and isinstance(payload["changed_files"], list)
    assert "diff_text" in payload and isinstance(payload["diff_text"], str)

    checks = payload.get("checks")
    assert isinstance(checks, dict)
    assert "pytest" in checks and "healthcheck" in checks
    assert checks["pytest"].get("skipped") is True
    assert checks["healthcheck"].get("skipped") is True
    # Optional checks should appear and default to skipped
    for opt in ("lint", "typecheck", "ui_build", "regression"):
        assert opt in checks
        assert checks[opt].get("skipped") is True
