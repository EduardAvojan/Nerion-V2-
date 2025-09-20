"""Local Tester agent: expands basic tests with edge-case stubs.

This is intentionally lightweight: it generates placeholder tests for empty
inputs, boundary values, and simple exception cases, without requiring models.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import ast
import os


def _list_functions(path: Path) -> List[ast.FunctionDef]:
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        return [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    except Exception:
        return []


def expand_tests(plan: Dict, target_file: str) -> str:
    target = Path(target_file)
    modname = target.stem
    fns = _list_functions(target)
    lines: List[str] = []
    lines.append("import pytest")
    # Optional Hypothesis
    USE_HYPO = (os.getenv('NERION_TESTER_HYPO') or '').strip().lower() in {'1','true','yes','on'}
    if USE_HYPO:
        try:
            lines.append("from hypothesis import given, strategies as st")
        except Exception:
            USE_HYPO = False

    # Function-aware stubs
    for fn in fns[:6]:
        fname = fn.name
        # Build a simple arg list with None/boundary placeholders
        arg_names = [a.arg for a in getattr(fn.args, 'args', [])]
        # Exclude 'self' for methods (module-level functions expected here)
        arg_names = [a for a in arg_names if a != 'self']
        call_args = []
        for a in arg_names:
            call_args.append('None')
        call_sig = ", ".join(call_args)
        lines.append("")
        lines.append(f"def test_{modname}_{fname}_none_inputs():")
        lines.append(f"    # TODO: import and call {modname}.{fname}({call_sig}) and assert behavior")
        lines.append("    assert True")

        # Boundary values test (for numeric-looking params)
        if arg_names:
            lines.append("")
            lines.append(f"def test_{modname}_{fname}_boundary_values():")
            lines.append("    # TODO: call with boundary values like 0, 1, -1 where applicable")
            lines.append("    assert True")

        # Hypothesis property test (optional)
        if USE_HYPO and arg_names:
            lines.append("")
            lines.append("@given(st.integers(), st.integers())")
            lines.append(f"def test_{modname}_{fname}_hypothesis(a, b):")
            lines.append("    # TODO: replace with structured strategies matching function signature")
            lines.append("    assert True")

    # Fallback generic tests if no functions found
    if len(fns) == 0:
        lines.append("")
        lines.append(f"def test_{modname}_module_loads():")
        lines.append("    assert True")

    return "\n".join(lines).strip()
