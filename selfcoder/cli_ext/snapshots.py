


from __future__ import annotations
import argparse
from selfcoder.vcs import git_ops

def _cmd_snapshot(args: argparse.Namespace) -> int:
    try:
        git_ops.snapshot(args.message)
        return 0
    except Exception as exc:
        print(f"snapshot failed: {exc}")
        return 1

def _cmd_restore(args: argparse.Namespace) -> int:
    try:
        git_ops.restore_snapshot(snapshot_ts=args.snapshot, files=args.files)
        return 0
    except Exception as exc:
        print(f"restore failed: {exc}")
        return 1

def register(subparsers) -> None:
    ss = subparsers.add_parser("snapshot", help="write a snapshot manifest")
    ss.add_argument("--message", default="Nerion snapshot")
    ss.set_defaults(func=_cmd_snapshot)

    sr = subparsers.add_parser("restore", help="restore files from a snapshot")
    sr.add_argument("--snapshot", dest="snapshot")
    sr.add_argument("--files", nargs="*", type=str)
    sr.set_defaults(func=_cmd_restore)