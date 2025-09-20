from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
import inspect

from selfcoder.actions import apply_actions_via_ast
from selfcoder.actions.crossfile import apply_crossfile_rename, RenameSpec, _apply_to_text
from ops.security import fs_guard

REPO_ROOT = Path(__file__).resolve().parents[1]

def _ensure_repo_path(p: str | Path) -> Path:
    """Resolve and enforce that the path is inside the repository root."""
    return fs_guard.ensure_in_repo(REPO_ROOT, str(p))


def _plan_from_request(request: str) -> List[Dict[str, Any]]:
    req = (request or "").lower()
    actions: List[Dict[str, Any]] = []
    if any(k in req for k in ["log", "logging", "instrument"]):
        actions.append({"kind": "inject_logging", "target_func": "main"})
    else:
        actions.append({"kind": "normalize_imports"})
    return actions


essay = """Nerion autocoder (minimal).\n\nThis command reads a Python file, derives a tiny action plan from\n`--request`, and applies those actions via AST. In `--dry-run` mode,\nit only parses and transforms in-memory to validate the pipeline.\n"""


def main(argv: List[str] | None = None) -> int:
    # Normalize argv in case the first token was passed as a single string with spaces
    # (some shells/scripts may pass "add logging" as one token).
    if argv is None:
        argv_list = sys.argv[1:]
    else:
        argv_list = list(argv)

    if argv_list and (" " in argv_list[0]):
        # Split the first token into separate args, e.g., "add logging" -> ["add", "logging"]
        head_parts = argv_list[0].split()
        argv_list = head_parts + argv_list[1:]

    parser = argparse.ArgumentParser(description=essay)
    subparsers = parser.add_subparsers(dest="command")

    # Default apply path (no subcommand)
    apply_parser = subparsers.add_parser("apply", help="Apply AST-based actions")
    apply_parser.add_argument("--request", required=True, help="What change to attempt (natural language)")
    apply_parser.add_argument("--file", required=True, help="Target Python file to analyze/patch")
    apply_parser.add_argument("--dry-run", action="store_true", help="Do not write changes; just simulate")

    # Legacy alias: `add logging ...` -> same as apply with request="logging ..."
    add_parser = subparsers.add_parser("add", help="Legacy alias for 'apply'")
    add_parser.add_argument("request_words", nargs="+", help="Words describing the change (legacy form)")
    add_parser.add_argument("--file", required=True, help="Target Python file to analyze/patch")
    add_parser.add_argument("--dry-run", action="store_true", help="Do not write changes; just simulate")

    # Rename subcommand
    rename_parser = subparsers.add_parser("rename", help="Apply crossfile rename")
    rename_parser.add_argument("--old-module", required=True, help="Old module name")
    rename_parser.add_argument("--old-attr", required=True, help="Old attribute name")
    rename_parser.add_argument("--new-module", required=True, help="New module name")
    rename_parser.add_argument("--new-attr", required=True, help="New attribute name")
    rename_parser.add_argument("--file", required=True, help="Target Python file to analyze/patch")
    rename_parser.add_argument("--dry-run", action="store_true", help="Do not write changes; just simulate")

    # Selfcheck subcommand
    subparsers.add_parser("selfcheck", help="Run self-checks for AST and crossfile rename")

    args = parser.parse_args(argv_list)

    if args.command == "add":
        # Convert legacy form to the standard apply path
        args.command = "apply"
        args.request = " ".join(args.request_words)

    if args.command == "rename":
        target = _ensure_repo_path(Path(args.file))
        specs = [
            RenameSpec(
                old_module=args.old_module,
                old_attr=args.old_attr,
                new_module=args.new_module,
                new_attr=args.new_attr,
            )
        ]

        if args.dry_run:
            # Read the file and apply the rename in-memory, print preview.
            source = target.read_text(encoding="utf-8")
            # _apply_to_text expects (text, specs)
            patched, edits = _apply_to_text(source, specs)
            print("=== [dry-run] Preview of changes ===")
            print(patched)
            return 0

        # Not dry-run: actually apply and write to disk.
        res = apply_crossfile_rename(specs, files=[str(target)], project_root=Path.cwd())
        source = target.read_text(encoding="utf-8")
        # Handle different return shapes.
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], dict):
            patched_map, changed_files = res
            patched = patched_map.get(str(target))
        elif isinstance(res, int):
            print(f"patched {res} file(s)")
            return 0
        elif isinstance(res, dict):
            patched = res.get(str(target))
        else:
            print("rename completed")
            return 0
        if patched is None or patched == source:
            print("no changes needed")
            return 0
        backup = target.with_suffix(target.suffix + ".bak")
        backup.write_text(source, encoding="utf-8")
        logging.warning("NERION.PATCH.APPLY %s -> caller=%s", __file__, inspect.stack()[1].function)
        target.write_text(patched, encoding="utf-8")
        print(f"patched {target} (backup: {backup.name})")
        return 0

    elif args.command == "selfcheck":
        # Run AST dry-run
        dummy_source = "def dummy_func():\n    pass\n"
        try:
            ast_actions = _plan_from_request("no-op")
            _ = apply_actions_via_ast(dummy_source, ast_actions)
        except Exception as e:
            print(f"AST selfcheck failed: {e}")
            return 1

        # Run crossfile rename dry-run
        try:
            specs = [RenameSpec(old_module="oldmod", old_attr="oldattr", new_module="newmod", new_attr="newattr")]
            _ = apply_crossfile_rename(specs, files=["app/nerion_chat.py"], project_root=Path.cwd())
        except Exception as e:
            print(f"Crossfile rename selfcheck failed: {e}")
            return 1

        print("selfcheck OK")
        return 0

    else:  # Default to apply path for backward compatibility
        target = _ensure_repo_path(Path(args.file))
        source = target.read_text(encoding="utf-8")

        actions = _plan_from_request(args.request)
        patched = apply_actions_via_ast(source, actions)

        if args.dry_run:
            # Successful transformation is all we need for health check
            print("dry-run OK")
            return 0

        if patched != source:
            backup = target.with_suffix(target.suffix + ".bak")
            backup.write_text(source, encoding="utf-8")
            logging.warning("NERION.PATCH.APPLY %s -> caller=%s", __file__, inspect.stack()[1].function)
            target.write_text(patched, encoding="utf-8")
            print(f"patched {target} (backup: {backup.name})")
        else:
            print("no changes needed")
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
