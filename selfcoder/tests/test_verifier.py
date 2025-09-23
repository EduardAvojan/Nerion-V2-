import sys
from pathlib import Path

import selfcoder.verifier as verifier
from selfcoder.verifier import failed_checks, run_post_apply_checks


def test_verifier_skips_when_no_commands(tmp_path, monkeypatch):
    for name in [
        "NERION_VERIFY_SMOKE_CMD",
        "NERION_VERIFY_INTEGRATION_CMD",
        "NERION_VERIFY_UI_CMD",
        "NERION_VERIFY_REG_CMD",
    ]:
        monkeypatch.delenv(name, raising=False)
    results = run_post_apply_checks(tmp_path)
    assert all(entry.get("skipped") for entry in results.values())
    assert not failed_checks(results)


def test_verifier_runs_configured_command(tmp_path, monkeypatch):
    script = tmp_path / "smoke_ok.py"
    script.write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    monkeypatch.setenv(
        "NERION_VERIFY_SMOKE_CMD",
        f"{sys.executable} {script}",
    )
    results = run_post_apply_checks(tmp_path)
    entry = results.get("smoke")
    assert entry is not None
    assert entry.get("rc") == 0
    assert not entry.get("skipped")
    assert not failed_checks(results)


def test_verifier_reports_failures(tmp_path, monkeypatch):
    script = tmp_path / "smoke_fail.py"
    script.write_text("import sys; sys.exit(5)\n", encoding="utf-8")
    monkeypatch.setenv(
        "NERION_VERIFY_SMOKE_CMD",
        f"{sys.executable} {script}",
    )
    results = run_post_apply_checks(tmp_path)
    assert failed_checks(results) == ["smoke"]


def test_verifier_default_command_detection(tmp_path, monkeypatch):
    for name in [
        "NERION_VERIFY_SMOKE_CMD",
        "NERION_VERIFY_INTEGRATION_CMD",
        "NERION_VERIFY_UI_CMD",
        "NERION_VERIFY_REG_CMD",
    ]:
        monkeypatch.delenv(name, raising=False)

    smoke_dir = tmp_path / "tests" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(verifier.shutil, "which", lambda cmd: sys.executable)

    recorded = {}

    def _fake_run(command, cwd, capture_output, text, timeout, check):
        recorded["cmd"] = command
        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Result()

    monkeypatch.setattr(verifier.subprocess, "run", _fake_run)

    results = run_post_apply_checks(tmp_path)
    entry = results["smoke"]
    assert not entry.get("skipped")
    assert entry.get("origin") == "default"
    assert recorded.get("cmd") == ["pytest", "tests/smoke", "-q"]
