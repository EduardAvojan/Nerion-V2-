
from __future__ import annotations
import argparse
from pathlib import Path
from selfcoder.orchestrator import run_actions_on_file

def _cmd(args: argparse.Namespace) -> int:
    target = Path(args.file)
    actions = []
    if args.module_doc:
        actions.append({"kind": "add_module_docstring", "payload": {"doc": args.module_doc}})
    if args.function and args.func_doc:
        actions.append({"kind": "add_function_docstring", "payload": {"function": args.function, "doc": args.func_doc}})
    elif args.function and not args.func_doc:
        print("error: --function requires --func-doc")
        return 2
    if not actions:
        print("nothing to do: provide --module-doc and/or --function with --func-doc")
        return 0
    try:
        run_actions_on_file(target, actions, dry_run=args.dry_run)
        if args.dry_run:
            print("[dry-run] docstring actions applied (no write)")
        return 0
    except Exception as exc:
        print(f"docstring action failed: {exc}")
        return 1

def register(subparsers) -> None:
    sd = subparsers.add_parser("docstring", help="apply docstrings via AST")
    sd.add_argument("--file", required=True)
    sd.add_argument("--module-doc", dest="module_doc")
    sd.add_argument("--function")
    sd.add_argument("--func-doc", dest="func_doc")
    sd.add_argument("--dry-run", action="store_true")
    sd.set_defaults(func=_cmd)

