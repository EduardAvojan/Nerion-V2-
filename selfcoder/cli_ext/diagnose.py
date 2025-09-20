


from __future__ import annotations

import argparse


def cmd_diagnose(args: argparse.Namespace) -> int:
    from selfcoder import diagnostics
    ok, text = diagnostics.run_diagnostics(
        json_output=getattr(args, "json", False),
        color=not getattr(args, "no_color", False),
    )
    print(text, end="")
    return 0 if ok else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("diagnose", help="run full system diagnostics")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_diagnose)