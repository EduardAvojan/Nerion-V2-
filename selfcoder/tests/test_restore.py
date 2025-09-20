

from pathlib import Path
import os
import io
import sys

import pytest

from selfcoder.vcs import git_ops


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_snapshot_and_restore_single_file(tmp_path, monkeypatch, capsys):
    # Work inside an isolated temp project
    monkeypatch.chdir(tmp_path)

    # Create a tiny project file structure
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")

    # Ensure SAFE_MODE is off so copies are written
    monkeypatch.setenv("SELFCODER_SAFE_MODE", "0")
    # Make sure no allow-only filter blocks our temp files
    monkeypatch.delenv("SELFCODER_ALLOW_ONLY", raising=False)
    # Avoid dry-run
    monkeypatch.delenv("SELFCODER_DRYRUN", raising=False)

    # Take a snapshot (should create backups/snapshots/... under tmp_path)
    git_ops.snapshot("pytest snapshot")

    # Mutate the file
    target.write_text("x = 2\n", encoding="utf-8")
    assert _read(target) == "x = 2\n"

    # Restore only this file from the latest snapshot
    git_ops.restore_snapshot(files=[Path("pkg/mod.py")])

    # Expect original content restored
    assert _read(target) == "x = 1\n"

    # Also assert the snapshot directories exist
    snaps_root = tmp_path / "backups" / "snapshots"
    assert snaps_root.exists()
    # There should be exactly one timestamped directory
    ts_dirs = [p for p in snaps_root.iterdir() if p.is_dir()]
    assert len(ts_dirs) == 1


def test_restore_snapshot_missing_ts_is_noop(tmp_path, monkeypatch, capsys):
    # Isolated working dir
    monkeypatch.chdir(tmp_path)

    # Create something
    p = tmp_path / "file.py"
    p.write_text("print('hi')\n", encoding="utf-8")

    # Try restoring a TS that doesn't exist — should not raise
    git_ops.restore_snapshot(snapshot_ts="19990101_000000", files=[Path("file.py")])

    # File should be unchanged
    assert _read(p) == "print('hi')\n"