from __future__ import annotations

import sys
from pathlib import Path

from selfcoder.cli_ext.patch import _apply_selected_hunks

try:
    from ops.security.safe_subprocess import safe_run
except Exception:  # pragma: no cover
    import subprocess

    def safe_run(argv, **kwargs):  # type: ignore
        return subprocess.run(
            argv,
            **{k: v for k, v in kwargs.items() if k in ("cwd", "timeout", "check", "capture_output", "text")}
        )


def test_apply_selected_hunks_basic():
    # Old vs New with two non-equal hunks: replace one line, then insert one line
    old = "a\nb\nc\n"
    new = "a\nB\nc\nD\n"

    # Select first hunk only (replace b -> B), not the insertion
    out1 = _apply_selected_hunks(old, new, [0])
    assert out1 == "a\nB\nc\n"

    # Select second hunk only (insert D), keep original b
    out2 = _apply_selected_hunks(old, new, [1])
    assert out2 == "a\nb\nc\nD\n"

    # Select both hunks gives the full new text
    out3 = _apply_selected_hunks(old, new, [0, 1])
    assert out3 == new


def test_minimal_subset_runner_pass(tmp_path: Path):
    # Create an isolated test project with a passing smoke test
    root = tmp_path / "proj"
    (root / "tests" / "smoke").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "smoke" / "test_smoke.py").write_text(
        """
def test_ok():
    assert 1 + 1 == 2
""".lstrip(),
        encoding="utf-8",
    )
    args = [sys.executable, "-m", "pytest", "-q", "-x", "tests/smoke"]
    res = safe_run(args, cwd=root, timeout=120, check=False, capture_output=True)
    assert res.returncode == 0


def test_minimal_subset_runner_fail_and_parse(tmp_path: Path):
    # Create an isolated test project with a failing test and verify failure code
    root = tmp_path / "proj2"
    (root / "tests").mkdir(parents=True, exist_ok=True)
    failing = root / "tests" / "test_fail.py"
    failing.write_text(
        """
def test_fail():
    assert 2 + 2 == 5
""".lstrip(),
        encoding="utf-8",
    )
    args = [sys.executable, "-m", "pytest", "-q", "-x"]
    res = safe_run(args, cwd=root, timeout=120, check=False, capture_output=True)
    # Non-zero rc indicates failure of subset
    assert res.returncode != 0
    def _to_str(b):
        if b is None:
            return ""
        return b.decode('utf-8', errors='ignore') if isinstance(b, (bytes, bytearray)) else str(b)
    out = _to_str(res.stdout) + "\n" + _to_str(res.stderr)
    # Best-effort: failing node id should be visible in stdout
    assert "tests/test_fail.py::test_fail" in out
