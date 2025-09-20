from __future__ import annotations

import os
from pathlib import Path

from selfcoder.repair import proposer as P
from selfcoder.cli_ext.bench import _score_candidate


def _write_task_with_tests(tmp: Path, mod_src: str, test_src: str) -> Path:
    task = tmp / "task1"
    (task / "tests").mkdir(parents=True, exist_ok=True)
    (task / "mod.py").write_text(mod_src, encoding="utf-8")
    (task / "tests" / "test_basic.py").write_text(test_src, encoding="utf-8")
    return task


def test_shadow_eval_import_fix(tmp_path: Path):
    # Failing task: mod.py uses xfoo without import; test asserts function returns 1
    mod_src = """
def run():
    return 1 if xfoo.ok else 0
""".lstrip()
    test_src = """
from mod import run

def test_run():
    assert run() == 1
""".lstrip()
    task = _write_task_with_tests(tmp_path, mod_src, test_src)
    # Build synthetic failure context
    ctx = {
        "_task_dir": str(task.resolve()),
        "failures": [
            {
                "traceback": "ModuleNotFoundError: No module named 'xfoo'",
                "frames": [{"file": str(task / 'mod.py'), "line": 2}],
            }
        ],
    }
    diffs = P.propose_diff_multi(ctx)
    assert diffs
    # Provide a local xfoo module so the import is resolvable
    (task / "xfoo.py").write_text("ok = True\n", encoding="utf-8")
    # Shadow-eval subset (subprocess pytest)
    ok, shadow = _score_candidate(task, diffs[0], ["tests/test_basic.py::test_run"])
    assert ok and shadow is not None


def test_shadow_eval_typeerror_fix(tmp_path: Path):
    # Failing task: add("2", 3) should be 5
    mod_src = """
def add(a, b):
    return a + b
""".lstrip()
    test_src = """
from mod import add

def test_add():
    assert add("2", 3) == 5
""".lstrip()
    task = _write_task_with_tests(tmp_path, mod_src, test_src)
    ctx = {
        "_task_dir": str(task.resolve()),
        "failures": [
            {
                "traceback": "TypeError: unsupported operand type(s) for +: 'str' and 'int'",
                "frames": [{"file": str(task / 'mod.py'), "line": 2}],
            }
        ],
    }
    diffs = P.propose_diff_multi(ctx)
    assert diffs
    ok, shadow = _score_candidate(task, diffs[0], ["tests/test_basic.py::test_add"])
    assert ok and shadow is not None


def test_shadow_eval_valueerror_fix(tmp_path: Path):
    # Failing task: parse("foo") should return None instead of raising
    mod_src = """
def parse(x):
    return int(x)
""".lstrip()
    test_src = """
from mod import parse

def test_parse():
    assert parse("foo") is None
""".lstrip()
    task = _write_task_with_tests(tmp_path, mod_src, test_src)
    ctx = {
        "_task_dir": str(task.resolve()),
        "failures": [
            {
                "traceback": "ValueError: invalid literal for int() with base 10: 'foo'",
                "frames": [{"file": str(task / 'mod.py'), "line": 2}],
            }
        ],
    }
    diffs = P.propose_diff_multi(ctx)
    assert diffs
    ok, shadow = _score_candidate(task, diffs[0], ["tests/test_basic.py::test_parse"])
    assert ok and shadow is not None
