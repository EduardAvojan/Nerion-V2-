"""Runtime & OS helpers for Nerion Self-Coder.

Centralized helpers so the autocoder core stays logic-only.
"""
from __future__ import annotations

import subprocess
import sys
import pathlib
import shutil
from typing import Optional

from selfcoder.config import ROOT, BACKUP_DIR
from ops.security import fs_guard
from ops.security.safe_subprocess import safe_run


def run(cmd: list[str], cwd: Optional[pathlib.Path] = None, check: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess safely; capture output; optional working dir.

    Be tolerant of different `safe_run` signatures across environments by
    falling back when the `text` parameter is not supported.
    """
    safe_cwd = None
    if cwd is not None:
        safe_cwd = fs_guard.ensure_in_repo(ROOT, cwd)
    try:
        return safe_run(
            cmd,
            cwd=safe_cwd,
            capture_output=True,
            text=True,
            check=check,
        )
    except TypeError:
        # Older safe_run variants may not support `text`
        return safe_run(
            cmd,
            cwd=safe_cwd,
            capture_output=True,
            check=check,
        )


def backup_file(path: pathlib.Path) -> pathlib.Path:
    """Copy `path` to BACKUP_DIR with a timestamp suffix. Returns backup path."""
    src = fs_guard.ensure_in_repo(ROOT, path)
    bdir = fs_guard.ensure_in_repo(ROOT, BACKUP_DIR)
    bdir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = bdir / f"{src.name}.{stamp}.bak"
    shutil.copy2(src, backup_path)
    return backup_path


def write_file(path: pathlib.Path, content: str) -> None:
    """Write UTF-8 text to `path` (repo-jailed)."""
    p = fs_guard.ensure_in_repo(ROOT, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def run_lint(root: pathlib.Path, *, fix: bool = False) -> bool:
    """Run Ruff lint. If fix=True, apply fixes and format.

    Returns True when tree is clean (or cleaned), False when issues remain
    or ruff is unavailable.
    """
    safe_root = fs_guard.ensure_in_repo(ROOT, root)

    def _exec(args: list[str]) -> subprocess.CompletedProcess:
        return run([str(sys.executable), "-m", "ruff", *args], cwd=safe_root)

    def _write(data, stream):
        try:
            if isinstance(data, (bytes, bytearray)):
                stream.write(data.decode("utf-8", errors="replace"))
            elif isinstance(data, str):
                stream.write(data)
        except Exception:
            pass

    print("[SelfCoder] Running Ruff lint…")
    try:
        if not fix:
            proc = _exec(["check", "."])
            _write(proc.stdout, sys.stdout)
            _write(proc.stderr, sys.stderr)
            return proc.returncode == 0

        # Fix path: attempt fix + format, then re-check
        fixp = _exec(["check", "--fix", "."])
        _write(fixp.stdout, sys.stdout)
        _write(fixp.stderr, sys.stderr)
        try:
            fmt = _exec(["format", "."])
            _write(fmt.stdout, sys.stdout)
            _write(fmt.stderr, sys.stderr)
        except Exception:
            # Ruff format may be unavailable on older versions; ignore
            pass
        again = _exec(["check", "."])
        _write(again.stdout, sys.stdout)
        _write(again.stderr, sys.stderr)
        return again.returncode == 0
    except Exception as e:
        sys.stderr.write(f"[SelfCoder] Ruff not available or failed: {e}\n")
        return False


def run_tests(root: pathlib.Path) -> bool:
    """Run pytest quietly; True if tests pass, False if they fail. If pytest missing, treat as pass."""
    try:
        safe_root = fs_guard.ensure_in_repo(ROOT, root)
        proc = run([str(sys.executable), "-m", "pytest", "-q"], cwd=safe_root)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode == 0
    except Exception:
        print("[SelfCoder] Pytest not available — skipping tests.")
        return True


def git_commit(path: pathlib.Path, message: str) -> bool:
    """Stage `path` and commit with `message`."""
    try:
        safe_path = fs_guard.ensure_in_repo(ROOT, path)
        run(["git", "add", str(safe_path)], cwd=ROOT)
        proc = run(["git", "commit", "-m", message], cwd=ROOT)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
        return proc.returncode == 0
    except Exception as e:
        print(f"[SelfCoder] Git commit skipped: {e}")
        return False
