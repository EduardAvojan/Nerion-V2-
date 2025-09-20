from __future__ import annotations

from pathlib import Path

from selfcoder.cli_ext.bench import _run_pytest_subset


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _make_flaky_task(root: Path) -> tuple[Path, list[str]]:
    """Create a tiny task with one flaky test that fails first run and passes on rerun.

    The test creates a flag file on first run and fails; reruns see the flag and pass.
    Returns (task_dir, subset_nodeids).
    """
    task = root / "flaky_subset_task"
    _write(task / "m.py", "def nop():\n    return None\n")
    test_code = (
        "import pathlib\n"
        "def test_flaky():\n"
        "    flag = pathlib.Path('FLAKY_FLAG')\n"
        "    if not flag.exists():\n"
        "        flag.write_text('1')\n"
        "        assert False\n"
        "    assert True\n"
    )
    _write(task / "tests" / "test_flaky.py", test_code)
    nodeids = ["tests/test_flaky.py::test_flaky"]
    return task, nodeids


def test_subset_flaky_retry(monkeypatch, tmp_path: Path):
    # Case 1: retry disabled → subset fails
    task1, nodes1 = _make_flaky_task(tmp_path)
    monkeypatch.delenv("NERION_BENCH_USE_LIBPYTEST", raising=False)
    monkeypatch.setenv("NERION_BENCH_FLAKY_RETRY", "0")
    ok1 = _run_pytest_subset(task1, nodes1)
    assert ok1 is False

    # Case 2: retry enabled → same flakiness pattern should pass within one call
    task2, nodes2 = _make_flaky_task(tmp_path)
    monkeypatch.setenv("NERION_BENCH_FLAKY_RETRY", "1")
    ok2 = _run_pytest_subset(task2, nodes2)
    assert ok2 is True

