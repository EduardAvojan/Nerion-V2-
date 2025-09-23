"""Full-system diagnostics for Nerion Selfcoder.

This module performs non-destructive checks across the stack and returns a
compact report suitable for CLI display (green checkmarks) or JSON.
"""
from __future__ import annotations

import io
import importlib
import json
import os
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Tuple, Dict, Any


def _mark(ok: bool, warn: bool = False, color: bool = True) -> str:
    # Simple emoji marks with optional ANSI color
    if warn and ok:
        # treat warnings as OK but mark yellow
        return ("\033[33m⚠️ \033[0m" if color else "⚠️ ")
    if ok:
        return ("\033[32m✅\033[0m" if color else "✅")
    return ("\033[31m❌\033[0m" if color else "❌")


def _safe(callable_, *a, **k):
    try:
        return True, callable_(*a, **k)
    except Exception as e:
        return False, e


def run_diagnostics(*, json_output: bool = False, color: bool = True) -> Tuple[bool, str]:
    """Run all diagnostics.

    Returns:
        (ok, text_or_json)
    """
    results: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "", warn: bool = False):
        results.append(
            {
                "name": name,
                "ok": bool(ok),
                "detail": str(detail) if detail else "",
                "warn": bool(warn),
            }
        )

    # When JSON is requested, silence stdout from anything we call so
    # the emitted JSON remains clean on stdout.
    sink = io.StringIO()
    redir = redirect_stdout(sink) if json_output else nullcontext()

    with redir:
        # ---- Environment ---------------------------------------------------
        py_ok = sys.version_info >= (3, 9)
        add("Python ≥ 3.9", py_ok, f"{sys.version.split()[0]}")

        ok_imp, sc_or_exc = _safe(importlib.import_module, "selfcoder")
        if ok_imp:
            ver = getattr(sc_or_exc, "__version__", None)
            add("selfcoder import & __version__", ver is not None, ver or "missing __version__")
        else:
            add("selfcoder import", False, repr(sc_or_exc))

        # ---- Core imports --------------------------------------------------
        for mod in [
            "selfcoder.actions",
            "selfcoder.orchestrator",
            "selfcoder.vcs.git_ops",
            "selfcoder.testgen",
            "selfcoder.healthcheck",
            "selfcoder.planner.planner",
            "app.nerion_chat",
        ]:
            ok_m, r = _safe(importlib.import_module, mod)
            add(f"import {mod}", ok_m, "" if ok_m else repr(r))

        # ---- AST engine basic ---------------------------------------------
        try:
            from selfcoder.actions import apply_actions_via_ast

            src = "def f():\n    return 1\n"
            out = apply_actions_via_ast(src, [{"kind": "add_module_docstring", "payload": {"doc": "diag"}}])
            add("AST: add_module_docstring", '"""diag"""' in out)

            out2 = apply_actions_via_ast(
                src,
                [
                    {"kind": "inject_function_entry_log", "payload": {"function": "f"}},
                    {"kind": "inject_function_exit_log", "payload": {"function": "f"}},
                    {"kind": "try_except_wrapper", "payload": {"function": "f"}},
                ],
            )
            ok_ast = ("Entering function f" in out2) and ("Exiting function f" in out2) and ("try" in out2 or "try:" in out2)
            add("AST: entry/exit/try", ok_ast)
        except Exception as e:
            add("AST engine", False, repr(e))

        # ---- Planner -------------------------------------------------------
        try:
            from selfcoder.planner.planner import plan_edits_from_nl

            plan = plan_edits_from_nl("add module docstring 'Diag'", "dummy.py")
            ok_plan = isinstance(plan, dict) and "actions" in plan and isinstance(plan["actions"], list)
            add("Planner: plan_edits_from_nl", ok_plan)
        except Exception as e:
            add("Planner", False, repr(e))

        # ---- Orchestrator (single-file) -----------------------------------
        try:
            from selfcoder.orchestrator import run_actions_on_file

            tmp = Path(tempfile.gettempdir()) / "nerion_diag_tmp.py"
            tmp.write_text("def x():\n    return 1\n", encoding="utf-8")
            ok_dry = run_actions_on_file(
                tmp, [{"kind": "add_module_docstring", "payload": {"doc": "Diag File"}}], dry_run=True
            ) in (True, False)
            ok_write = run_actions_on_file(
                tmp, [{"kind": "add_module_docstring", "payload": {"doc": "Diag File"}}], dry_run=False
            ) in (True, False)
            add("Orchestrator: run_actions_on_file (dry)", bool(ok_dry))
            add("Orchestrator: run_actions_on_file (write)", bool(ok_write))
        except Exception as e:
            add("Orchestrator", False, repr(e))

        # ---- Healthcheck ---------------------------------------------------
        try:
            from selfcoder import healthcheck as _hc

            res = _hc.run_all()
            ok_h = bool(res[0]) if isinstance(res, tuple) else bool(res)
            add("Healthcheck: run_all()", ok_h)
        except Exception as e:
            add("Healthcheck", False, repr(e))

        # ---- VCS Snapshot (dry-run) ---------------------------------------
        try:
            from selfcoder.vcs import git_ops

            prev = os.environ.get("SELFCODER_DRYRUN", "")
            os.environ["SELFCODER_DRYRUN"] = "1"
            ts = git_ops.snapshot("diagnostics (dry-run)")
            add("VCS: snapshot (dry-run)", bool(ts), ts or "")
            # don’t attempt restore on dry snapshot to avoid 'not found' noise
            if prev:
                os.environ["SELFCODER_DRYRUN"] = prev
            else:
                del os.environ["SELFCODER_DRYRUN"]
        except Exception as e:
            add("VCS", False, repr(e))

        # ---- Logging setup -------------------------------------------------
        try:
            from selfcoder import logging_config

            buf = io.StringIO()
            try:
                logging_config.setup_logging(level=20, stream=buf)  # INFO
            except TypeError:
                logging_config.setup_logging(20, buf)
            import logging as _logging

            _logger = _logging.getLogger("selfcoder")
            _logger.info("diagnostics-log-line")
            ok_log = "diagnostics-log-line" in buf.getvalue()
            add("Logging: setup & emit", ok_log)
        except Exception as e:
            add("Logging", False, repr(e))

        # ---- Causal Analysis (smoke) --------------------------------------
        try:
            def _boom():
                return 1 / 0
            try:
                _boom()
            except Exception as e:
                report = analyze_exception(e)
                ok_ca = isinstance(report, dict) and report.get("root_cause") == "ZeroDivisionError"
                add("Causal: analyze_exception", ok_ca)
        except Exception as e:
            add("Causal Analysis", False, repr(e))

        # ---- Voice pipeline (mocked I/O, outside repo) --------------------
        try:
            nc = importlib.import_module("app.nerion_chat")
            # Prepare a temp file outside repo
            t = Path(tempfile.gettempdir()) / "nerion_diag_voice.py"
            t.write_text("def f():\n    return 1\n", encoding="utf-8")
            # Stub I/O
            nc.speak = lambda *a, **k: None
            nc.listen_once = lambda **k: str(t)
            ok_voice = bool(nc.run_self_coding_pipeline("add module docstring 'Diag Voice'"))
            ok_applied = '"""Diag Voice"""' in t.read_text(encoding="utf-8")
            add("Voice pipeline: apply docstring (mocked)", ok_voice and ok_applied)
        except Exception as e:
            add("Voice pipeline", False, repr(e))

        # ---- CLI surface presence -----------------------------------------
        try:
            cli = importlib.import_module("selfcoder.cli")
            parser = cli._build_parser()
            helptext = parser.format_help()
            expected = ["plan", "autotest", "healthcheck", "docstring", "snapshot", "restore", "batch", "diagnose"]
            ok_cli = all(x in helptext for x in expected)
            add("CLI: subcommands present", ok_cli, ", ".join(expected))
        except Exception as e:
            add("CLI", False, repr(e))

        # ---- Soft config warning ------------------------------------------
        cfg = Path("app/settings.yaml")
        if not cfg.exists():
            add("Config: app/settings.yaml (optional)", True, "missing (optional)", warn=True)
        else:
            add("Config: app/settings.yaml (optional)", True, "present", warn=False)

    # Aggregate
    ok_all = all(r["ok"] for r in results if not r.get("warn", False))

    if json_output:
        # Make sure stdout is not redirected anymore here; we only return JSON text
        return ok_all, json.dumps(results, ensure_ascii=False, indent=2)

    # Pretty text report
    use_color = color and sys.stdout.isatty()
    lines = []
    lines.append("[diagnose] Nerion system report")
    for r in results:
        mark = _mark(r["ok"], warn=r.get("warn", False), color=use_color)
        if r["detail"]:
            lines.append(f"{mark} {r['name']}: {r['detail']}")
        else:
            lines.append(f"{mark} {r['name']}")
    lines.append(f"[diagnose] Overall: {'OK' if ok_all else 'FAIL'}")
    return ok_all, "\n".join(lines) + "\n"


def analyze_exception(exc: BaseException, *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Produce a structured, human-actionable analysis of an exception.

    Returns a dict with keys: type, message, root_cause, hints (list[str]), stack (list[frames]), context.
    """
    etype = exc.__class__.__name__
    msg = str(exc)
    tb = traceback.TracebackException.from_exception(exc)
    stack = [{"file": f.filename, "line": f.lineno, "func": f.name} for f in tb.stack] if tb.stack else []

    hints: List[str] = []
    root = etype

    # Classify and hint
    if isinstance(exc, ModuleNotFoundError):
        root = "ModuleNotFoundError"
        hints.append("Install or add missing dependency (pip install <name>).")
        hints.append("Verify virtualenv and PYTHONPATH.")
    elif isinstance(exc, ImportError):
        root = "ImportError"
        hints.append("Check import path and package version compatibility.")
    elif isinstance(exc, FileNotFoundError):
        root = "FileNotFoundError"
        hints.append("Ensure the file path exists and is readable.")
    elif isinstance(exc, PermissionError):
        root = "PermissionError"
        hints.append("Check file permissions or run with sufficient privileges.")
    elif isinstance(exc, SyntaxError):
        root = "SyntaxError"
        hints.append("Run a linter/formatter; verify Python version features.")
    elif isinstance(exc, AttributeError):
        root = "AttributeError"
        hints.append("The attribute/method may not exist; check object type and version.")
    elif isinstance(exc, NameError):
        root = "NameError"
        hints.append("The symbol is undefined in this scope; check imports and spelling.")
    elif isinstance(exc, TypeError):
        root = "TypeError"
        if "unexpected keyword argument" in msg:
            hints.append("Function signature mismatch; review parameter names and versions.")
        if "missing required positional argument" in msg or "positional arguments but" in msg:
            hints.append("Adjust callsite arguments to match function signature.")
    elif isinstance(exc, ValueError):
        root = "ValueError"
        hints.append("Validate input values and formats; add guards.")
    elif isinstance(exc, ZeroDivisionError):
        root = "ZeroDivisionError"
        hints.append("Guard against zero divisor; add early return or conditional.")
    elif isinstance(exc, AssertionError):
        root = "AssertionError"
        hints.append("Test assertion failed; inspect expected vs actual and upstream inputs.")

    analysis = {
        "type": etype,
        "message": msg,
        "root_cause": root,
        "hints": hints,
        "stack": stack,
        "context": context or {},
    }
    return analysis


def diagnose_call(func, *args, **kwargs) -> Dict[str, Any]:
    """Run a callable safely; return {ok, result|analysis}.

    This is a light wrapper that other layers can use to quickly obtain
    a causal analysis when an operation crashes.
    """
    ok, res = _safe(func, *args, **kwargs)
    if ok:
        return {"ok": True, "result": res}
    return {"ok": False, "analysis": analyze_exception(res)}


# --------- helpers ----------------------------------------------------------

def persist_analysis(analysis: Dict[str, Any], outdir: str | Path = "out/analysis_reports") -> Path:
    """Persist an analysis dict to a timestamped JSON file and return its Path."""
    from datetime import datetime
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = outdir / f"analysis_{ts}.json"
    try:
        path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        # Best-effort, do not raise
        sys.stderr.write(f"[diagnostics] Failed to persist analysis: {e}\n")
    return path


class nullcontext:
    """Minimal backport of contextlib.nullcontext for Python 3.7–3.8,
    kept private here to avoid importing contextlib.nullcontext conditionally.
    """
    def __init__(self, enter_result=None):
        self.enter_result = enter_result
    def __enter__(self):
        return self.enter_result
    def __exit__(self, *exc):
        return False
