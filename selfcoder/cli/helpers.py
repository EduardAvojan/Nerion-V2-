"""CLI helper utilities."""
from __future__ import annotations

import sys

try:
    from selfcoder.vcs import git_ops
except Exception:
    class _GitOpsFallback:
        @staticmethod
        def snapshot(*_a, **_k):
            return "0"
        @staticmethod
        def restore_snapshot(*_a, **_k):
            return None
    git_ops = _GitOpsFallback()


def positive_exit(ok: bool) -> int:
    """Convert boolean to exit code: 0 for True, 1 for False."""
    return 0 if ok else 1


def apply_with_rollback(snapshot_message: str, apply_fn, check_fn=None) -> bool:
    """
    Take a VCS snapshot, call apply_fn(), then optionally call check_fn().
    If apply_fn raises or check_fn returns False, restore the snapshot and return False.
    Returns True on success.
    """
    ts = git_ops.snapshot(snapshot_message)
    # Normalize snapshot token to a string for restore_snapshot
    if isinstance(ts, (list, tuple)):
        ts = ts[0] if ts else ""
    elif isinstance(ts, dict):
        ts = ts.get("ts") or ts.get("timestamp") or next(iter(ts.values()), None)
    ts = str(ts)
    try:
        _ok_apply = bool(apply_fn())
    except Exception as exc:
        print(f"[apply] failed during apply: {exc}", file=sys.stderr)
        git_ops.restore_snapshot(snapshot_ts=ts)
        return False

    ok = True
    if check_fn is not None:
        try:
            ok = bool(check_fn())
        except Exception as exc:
            ok = False
            print(f"[check] failed to run: {exc}", file=sys.stderr)

    if not ok:
        git_ops.restore_snapshot(snapshot_ts=ts)
    return ok
