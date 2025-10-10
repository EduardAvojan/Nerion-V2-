"""Nerion Selfcoder CLI - Main entry point.

Commands:
  • plan         – plan edits from natural language (optional --apply to execute)
  • healthcheck  – run internal health checks
  • docstring    – add module/function docstrings via AST pipeline
  • snapshot     – write a snapshot manifest using VCS helper
  • diagnose     – run full system diagnostics
  • simulate     - run a command in a temporary shadow copy of the repo

This module exposes `main(argv=None)` for testability and `console_entry()`
for the console-script entry point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .parser_builder import build_parser

# Backward compatibility exports
from .parser_builder import build_parser as _build_parser
from .helpers import positive_exit as _positive_exit, apply_with_rollback as _apply_with_rollback
from .simulation import simulate_fallback as _simulate_fallback, maybe_simulate as _maybe_simulate
from .fallbacks import (
    register_plan_fallback as _register_plan_fallback,
    register_health_fallback as _register_health_fallback,
    register_rename_fallback as _register_rename_fallback,
)
from .autotest import cmd_autotest

# Import for test compatibility (tests may monkeypatch this)
from selfcoder.planner.prioritizer import build_planner_context


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    # Add small built-in subcommands: init, help (docs/help)
    sub = None
    try:
        sub = next(sp for sp in parser._subparsers._group_actions if isinstance(sp, argparse._SubParsersAction))
    except Exception:
        sub = None
    if sub is not None:
        # init
        p_init = sub.add_parser('init', help='write default policy and settings stubs if missing')
        def _run_init(_args):
            try:
                from selfcoder.cli_init import main as _init
                return int(_init([]))
            except Exception as e:
                print(f"[init] failed: {e}", file=sys.stderr)
                return 1
        p_init.set_defaults(func=_run_init)
        # help
        p_help = sub.add_parser('help', help='show a micro-guide from docs/help/<topic>.md')
        p_help.add_argument('topic')
        def _run_help(a):
            topic = (getattr(a, 'topic', '') or '').strip()
            if not topic:
                print('usage: nerion help <topic>')
                return 1
            path = Path('docs')/ 'help' / f'{topic}.md'
            if not path.exists():
                print(f"[help] no topic: {topic} (expected {path})")
                return 1
            try:
                print(path.read_text(encoding='utf-8'))
            except Exception as e:
                print(f"[help] failed to read: {e}")
                return 1
            return 0
        p_help.set_defaults(func=_run_help)
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        try:
            from app.version import BUILD_TAG
            print(BUILD_TAG)
        except Exception:
            print("unknown")
        return 0
    if hasattr(args, 'func'):
        try:
            return int(args.func(args))
        except SystemExit:
            raise
        except Exception as e:
            try:
                from core.ui.messages import fmt as _fmt_msg, Result as _MsgRes
                print(_fmt_msg('cli', 'error', _MsgRes.ERROR, str(e)))
            except Exception:
                print(f"[cli] error: {e}")
            return 2
    return 1


def console_entry() -> None:
    """Console script entry point."""
    sys.exit(main())


__all__ = [
    "main",
    "console_entry",
    "cmd_autotest",
    "_build_parser",
    "_positive_exit",
    "_apply_with_rollback",
    "_simulate_fallback",
    "_maybe_simulate",
    "_register_plan_fallback",
    "_register_health_fallback",
    "_register_rename_fallback",
]
