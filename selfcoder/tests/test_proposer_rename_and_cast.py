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


def test_nameerror_safe_rename_line(tmp_path: Path):
    f = tmp_path / "typo.py"
    f.write_text(
        """
def add(value):
    vlaue = value
    return vlaue + 1
""".lstrip(),
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path, f, 3, "NameError: name 'vlaue' is not defined")
    diffs = P.propose_diff_multi(ctx)
    patch = "\n".join(diffs)
    assert "return value + 1" in patch


def test_valueerror_cast_guard_inserts_try_except(tmp_path: Path):
    f = tmp_path / "cast.py"
    f.write_text(
        """
def parse(x):
    return int(x)
""".lstrip(),
        encoding="utf-8",
    )
    tb = "ValueError: invalid literal for int() with base 10: 'a'"
    ctx = _ctx(tmp_path, f, 2, tb)
    diffs = P.propose_diff_multi(ctx)
    patch = "\n".join(diffs)
    assert "try:" in patch and "except (ValueError, TypeError):" in patch


def test_typeerror_binary_op_guard_inserts_coercion(tmp_path: Path):
    f = tmp_path / "ops.py"
    f.write_text(
        """
def sum2(a,b):
    return a + b
""".lstrip(),
        encoding="utf-8",
    )
    tb = "TypeError: unsupported operand type(s) for +"
    ctx = _ctx(tmp_path, f, 2, tb)
    diffs = P.propose_diff_multi(ctx)
    patch = "\n".join(diffs)
    assert "if isinstance(a, str):" in patch and "if isinstance(b, str):" in patch

