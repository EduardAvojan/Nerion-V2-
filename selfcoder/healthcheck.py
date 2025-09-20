from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple
import sys
from ops.security.safe_subprocess import safe_run

from selfcoder.logging_config import setup_logging
from selfcoder.actions import apply_actions_via_ast
from selfcoder.orchestrator import (
    run_actions_on_file,
    run_ast_actions,
)
from selfcoder.vcs.git_ops import should_skip

# Optional coverage support
try:
    from selfcoder import coverage_utils as _covu  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _covu = None  # graceful fallback if not available


def _check_safe_subprocess() -> Tuple[bool, str]:
    try:
        result = safe_run([sys.executable, "-c", "print('hc')"])
        ok = b"hc" in result.stdout
        return ok, "safe_subprocess basic run"
    except Exception as e:
        return False, f"safe_subprocess failed: {e}"


def _check_logging() -> Tuple[bool, str]:
    stream = io.StringIO()
    logger = setup_logging(level=logging.INFO, stream=stream)
    logger.info("healthcheck-log-test")
    ok = "healthcheck-log-test" in stream.getvalue()
    return ok, "logging setup & emit"


def _check_actions_roundtrip() -> Tuple[bool, str]:
    src = "def f():\n    return 1\n"
    out = apply_actions_via_ast(src, [
        {"kind": "add_module_docstring", "payload": {"doc": "HC module"}},
        {"kind": "add_function_docstring", "payload": {"function": "f", "doc": "HC func"}},
        {"kind": "inject_function_entry_log", "payload": {"function": "f"}},
    ])
    ok = (
        '"""HC module"""' in out
        and '"""HC func"""' in out
        and "logger = logging.getLogger(__name__)" in out
        and "logger.info('Entering function f')" in out
    )
    return ok, "AST actions (docstrings + entry log)"


def _check_orchestrator_api() -> Tuple[bool, str]:
    # legacy helpers present via import above; also do a quick transform
    src = "def g():\n    return 2\n"
    patched = run_ast_actions(src, [
        {"kind": "add_module_docstring", "payload": {"doc": "Y"}},
    ])
    ok = '"""Y"""' in patched
    return ok, "orchestrator legacy API & run_ast_actions"


def _check_run_actions_on_file() -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mod.py"
        p.write_text("def h():\n    return 3\n", encoding="utf-8")
        changed = run_actions_on_file(
            p,
            [{"kind": "add_module_docstring", "payload": {"doc": "Z"}}],
            dry_run=False,
        )
        out = p.read_text(encoding="utf-8")
        ok = changed and '"""Z"""' in out
        return ok, "file-level AST action (write)"


def _check_should_skip() -> Tuple[bool, str]:
    # Hard-ignored example
    ok1 = should_skip(Path(".venv/lib/site.py"))[0] is True
    # A typical project file should not be skipped
    ok2 = should_skip(Path("selfcoder/healthcheck.py"))[0] is False
    return (ok1 and ok2), "vcs.should_skip hard-ignore + allow"


def _check_main_insert_helpers() -> Tuple[bool, str]:
    # Call the internal helpers directly with a temp file (no repo writes)
    from selfcoder.main import _insert_function_docstring as ins_fn
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tmp_mod.py"
        p.write_text("def k():\n    return 0\n", encoding="utf-8")
        ins_fn(p, "k", "temp doc", dry_run=False)
        txt = p.read_text(encoding="utf-8")
        ok = '"""temp doc"""' in txt
        return ok, "main.py docstring inserter (function)"



def _check_coverage_baseline() -> Tuple[bool, str]:
    """
    Report coverage baseline status (non-fatal).
    Returns (ok: bool, msg: str).
    """
    if _covu is None:
        return (True, "coverage: utils not available — skipping")

    try:
        baseline = _covu.load_baseline()
    except Exception as e:  # defensive
        return (True, f"coverage: failed to read baseline ({e!r}) — skipping")

    if not baseline:
        return (True, "coverage: no baseline saved yet — skipping")

    try:
        pct, _ = _covu.compare_to_baseline(baseline, baseline)
    except Exception as e:  # defensive
        return (True, f"coverage: baseline present (compare noop failed: {e!r})")

    return (True, f"coverage: baseline present ({pct:.1f}%)")


def _check_coverage_regression() -> Tuple[bool, str]:
    """
    If last-run coverage exists, compare to baseline.
    By default, informational only; set NERION_FAIL_ON_COVERAGE_DROP=1 to fail on regression.
    """
    if _covu is None:
        return (True, "coverage: utils not available — skipping")

    try:
        baseline = _covu.load_baseline()
        last = _covu.load_last_run()
    except Exception as e:  # defensive
        return (True, f"coverage: failed to read data ({e!r}) — skipping")

    if not baseline or not last:
        return (True, "coverage: baseline/last-run missing — skipping")

    try:
        current_pct, delta = _covu.compare_to_baseline(last, baseline)
    except Exception as e:  # defensive
        return (True, f"coverage: compare failed ({e!r}) — skipping")

    msg = f"coverage: last {current_pct:.1f}% ({delta:+.1f} vs baseline)"
    # Enforce failure on regression if env knob is set
    fail_on_drop = os.environ.get("NERION_FAIL_ON_COVERAGE_DROP", "").strip() in {"1", "true", "yes"}
    if fail_on_drop and delta < 0:
        return (False, msg + " — FAIL on drop")
    return (True, msg)


def run_all(verbose: bool = False) -> bool:
    checks = [
        _check_logging,
        _check_actions_roundtrip,
        _check_orchestrator_api,
        _check_run_actions_on_file,
        _check_should_skip,
        _check_main_insert_helpers,
        _check_coverage_baseline,
        _check_coverage_regression,
        _check_safe_subprocess,
    ]
    results: List[Tuple[bool, str]] = [c() for c in checks]
    if verbose:
        for ok, name in results:
            print(("✔" if ok else "✘"), name)
    return all(ok for ok, _ in results)


def run_healthcheck(verbose: bool = False) -> bool:
    """Compatibility alias used by self_improve.py; forwards to run_all."""
    return run_all(verbose=verbose)


__all__ = ["run_all", "run_healthcheck"]
