from __future__ import annotations

import argparse
from pathlib import Path

from selfcoder.io.runtime import run_lint
from core.ui.progress import progress
from core.ui.messages import fmt as _fmt_msg
from core.ui.messages import Result as _MsgRes


def _cmd_lint(args: argparse.Namespace) -> int:
    root = Path(getattr(args, "root", "."))
    with progress("lint"):
        ok = run_lint(root, fix=bool(getattr(args, "fix", False)))
    if not ok and not getattr(args, "fix", False):
        print("[lint] issues found. Run `nerion lint --fix` to attempt fixes.")
    print(_fmt_msg('lint', 'summary', _MsgRes.OK if ok else _MsgRes.FAIL, f"root={root}"))
    return 0 if ok else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("lint", help="run Ruff lint; add --fix to auto-fix and format")
    p.add_argument("--root", default=".", help="project root (default: '.')")
    p.add_argument("--fix", action="store_true", help="apply fixes and format, then re-run lint")
    p.set_defaults(func=_cmd_lint)
