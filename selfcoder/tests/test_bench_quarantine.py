from __future__ import annotations

from pathlib import Path

from selfcoder.cli_ext.bench import _run_pytest_all


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _make_flaky_task(root: Path) -> Path:
    task = root / "flaky_full_task"
    _write(task / "m.py", "def nop():\n    return None\n")
    test_code = (
        "import pathlib\n"
        "def test_flaky_full():\n"
        "    flag = pathlib.Path('FLAKY_FULL_FLAG')\n"
        "    if not flag.exists():\n"
        "        flag.write_text('1')\n"
        "        assert False\n"
        "    assert True\n"
    )
    _write(task / "tests" / "test_flaky_full.py", test_code)
    return task


def test_full_quarantine_reruns(monkeypatch, tmp_path: Path):
    # Case 1: quarantine disabled → full run fails
    task1 = _make_flaky_task(tmp_path)
    monkeypatch.delenv("NERION_BENCH_USE_LIBPYTEST", raising=False)
    # Explicitly disable quarantine reruns
    monkeypatch.setenv("NERION_BENCH_QUARANTINE_RERUNS", "0")
    ok1 = _run_pytest_all(task1)
    assert ok1 is False

    # Case 2: quarantine enabled → passes after rerun of the flaky test
    task2 = _make_flaky_task(tmp_path)
    monkeypatch.setenv("NERION_BENCH_QUARANTINE_RERUNS", "1")
    monkeypatch.setenv("NERION_BENCH_QUARANTINE_REQ_OK", "1")
    ok2 = _run_pytest_all(task2)
    assert ok2 is True
