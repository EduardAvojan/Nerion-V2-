import subprocess
import sys
from pathlib import Path
import os
import shutil

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
    shutil.copy(PROJECT_ROOT / "selfcoder/cli.py", selfcoder_dir / "cli.py")
    shutil.copy(PROJECT_ROOT / "selfcoder/simulation.py", selfcoder_dir / "simulation.py")
    shutil.copy(PROJECT_ROOT / "selfcoder/actions/crossfile.py", selfcoder_dir / "actions" / "crossfile.py")

    # Create dummy modules that are imported by the CLI
    (selfcoder_dir / "healthcheck.py").write_text("def run_all(): return (True, 'OK')")
    (selfcoder_dir / "orchestrator.py").write_text("def run_actions_on_file(*a, **kw): return True")
    (selfcoder_dir / "planner" / "planner.py").write_text("def plan_edits_from_nl(*a, **kw): return {'actions': [], 'target_file': 'app.py'}")
    (selfcoder_dir / "vcs" / "git_ops.py").write_text("def snapshot(*a, **kw): return 'ts'\ndef restore_snapshot(*a, **kw): pass")

    return project_dir

def run_cli_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Helper to run a CLI command with the correct environment."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
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
    stdout = result.stdout.strip()
    # Find the last occurrence of a JSON object by locating the last '{'
    last_open = stdout.rfind('{')
    assert last_open != -1, f"Expected JSON in output, got: {stdout[:200]}"
    json_blob = stdout[last_open:]

    import json as _json
    payload = _json.loads(json_blob)

    # Basic shape checks
    assert isinstance(payload, dict)
    assert "shadow_dir" in payload and isinstance(payload["shadow_dir"], str)
    assert payload.get("cmd_exit") == 0
    # Skipped checks should be None in JSON
    assert payload.get("pytest_exit") is None
    assert payload.get("healthcheck_exit") is None
    assert "changed" in payload and isinstance(payload["changed"], bool)
    assert "changed_files" in payload and isinstance(payload["changed_files"], list)
    assert "diff_text" in payload and isinstance(payload["diff_text"], str)