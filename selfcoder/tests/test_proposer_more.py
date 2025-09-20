from __future__ import annotations

from pathlib import Path
from selfcoder.repair import proposer as P


def _ctx(tmp: Path, file: Path, line_no: int, tb: str):
    return {
        "_task_dir": str(tmp.resolve()),
        "failures": [
            {
                "traceback": tb,
                "frames": [{"file": str(file), "line": line_no}],
            }
        ],
    }


def test_none_guard_inserts_in_function(tmp_path: Path):
    f = tmp_path / "mod.py"
    f.write_text(
        """
def foo(s):
    return s.lower()
""".lstrip(),
        encoding="utf-8",
    )
    # Error: NoneType has no attribute lower at the return line
    ctx = _ctx(tmp_path, f, 2, "AttributeError: 'NoneType' object has no attribute 'lower'")
    diffs = P.propose_diff_multi(ctx)
    patch = "\n".join(diffs)
    assert "if s is None:" in patch
    assert "return None" in patch


def test_index_guard_simple(tmp_path: Path):
    f = tmp_path / "arr.py"
    f.write_text(
        """
def get(arr, i):
    return arr[i]
""".lstrip(),
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path, f, 2, "IndexError: list index out of range")
    diffs = P.propose_diff_multi(ctx)
    patch = "\n".join(diffs)
    assert "if not (0 <= int(i) < len(arr)):" in patch
    assert "return None" in patch


def test_numeric_tolerance_fix(tmp_path: Path):
    f = tmp_path / "calc.py"
    # Failing line compares floats directly
    f.write_text(
        """
def ok(a, b):
    if a == b:
        return True
    return False
""".lstrip(),
        encoding="utf-8",
    )
    ctx = {
        "_task_dir": str(tmp_path.resolve()),
        "failures": [
            {
                "traceback": "AssertionError: floats are not equal",
                "frames": [{"file": str(f), "line": 2}],
            }
        ],
    }
    diffs = P.propose_diff_multi(ctx)
    patch = "\n".join(diffs)
    assert "math.isclose(" in patch
    assert "import math" in patch
