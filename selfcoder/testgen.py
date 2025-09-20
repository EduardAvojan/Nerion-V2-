

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List
from ops.security.safe_subprocess import safe_run
from ops.security import fs_guard
import sys
import datetime


SUPPORTED_ACTIONS = {
    "add_module_docstring",
    "add_function_docstring",
    "inject_function_entry_log",
    "inject_function_exit_log",
    "try_except_wrapper",
}


def generate_test_code(plan: Dict[str, Any], target_file: str) -> str:
    """
    Generate pytest code as a string that verifies the effects of the given plan
    on the specified target file.
    """
    actions = plan.get("actions", [])
    checks: List[str] = []

    for action in actions:
        kind = action.get("kind")
        payload = action.get("payload", {}) or {}
        if kind not in SUPPORTED_ACTIONS:
            continue

        if kind == "add_module_docstring":
            doc = payload.get("doc", "")
            checks.append(f"assert '\"\"\"{doc}\"\"\"' in src or \"'''{doc}'''\" in src")

        elif kind == "add_function_docstring":
            fn = payload.get("function")
            doc = payload.get("doc", "")
            checks.append(
                dedent(f"""
                assert f'def {fn}(' in src
                assert '\"\"\"{doc}\"\"\"' in src or \"'''{doc}'''\" in src
                """).strip()
            )

        elif kind == "inject_function_entry_log":
            fn = payload.get("function")
            checks.append(
                f"assert \"logger.info('Entering function {fn}')\" in src"
            )

        elif kind == "inject_function_exit_log":
            fn = payload.get("function")
            checks.append(
                f"assert \"logger.info('Exiting function {fn}')\" in src"
            )

        elif kind == "try_except_wrapper":
            fn = payload.get("function")
            checks.append(
                dedent(f"""
                assert f'def {fn}(' in src
                assert 'try:' in src
                assert 'except Exception as e:' in src
                """).strip()
            )

    if not checks:
        raise ValueError("No supported actions found in plan")

    checks_code = "\n    ".join(checks)

    return dedent(f"""
    import pytest
    from pathlib import Path

    def test_generated_plan_applied():
        src = Path(r\"\"\"{target_file}\"\"\").read_text(encoding="utf-8")
        {checks_code}
    """).strip()


def write_test_file_timestamp(plan: Dict[str, Any], target_file: str, output_dir: Path) -> Path:
    """
    Write the generated pytest code to a new file in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dir = fs_guard.ensure_in_repo(Path('.'), str(output_dir))
    test_path = safe_dir / f"test_plan_{ts}.py"
    code = generate_test_code(plan, target_file)
    test_path.write_text(code, encoding="utf-8")
    return test_path


def run_test_file(test_path: Path) -> int:
    """
    Run pytest on the given test file and return the pytest exit code.
    """
    p = fs_guard.ensure_in_repo(Path('.'), str(test_path))
    res = safe_run([sys.executable, "-m", "pytest", "-q", str(p)])
    return res.returncode


# --- Compatibility wrappers for CLI ---
def generate_tests_for_plan(plan, target_path):
    """
    Compatibility wrapper expected by CLI:
    returns pytest code (str) for a given plan and target file.
    """
    return generate_test_code(plan, str(target_path))


def write_test_file(code: str, out_path):
    """
    Compatibility wrapper expected by CLI:
    write provided pytest code to the given path.
    """
    safe_p = fs_guard.ensure_in_repo(Path('.'), str(out_path))
    safe_p.parent.mkdir(parents=True, exist_ok=True)
    safe_p.write_text(code, encoding="utf-8")
    return safe_p


def run_pytest_on_paths(paths):
    """
    Compatibility wrapper expected by CLI:
    invoke pytest on a list of file/directory paths and return exit code.
    """
    try:
        import pytest  # type: ignore
    except Exception:
        # Fallback to subprocess if pytest import fails
        import sys
        from ops.security.safe_subprocess import safe_run
        res = safe_run([sys.executable, "-m", "pytest", "-q", *[str(x) for x in paths]])
        return res.returncode
    return pytest.main([str(x) for x in paths])
