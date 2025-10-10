from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from ops.security.fs_guard import ensure_in_repo_auto

from selfcoder import healthcheck
import os

# We reference the top-level CLI helpers for simulation and rollback.
try:
    # These are defined in selfcoder/cli package
    from selfcoder.cli import _maybe_simulate, _apply_with_rollback  # type: ignore
except Exception:  # pragma: no cover - defensive import
    _maybe_simulate = None
    _apply_with_rollback = None


def cmd_rename(args: argparse.Namespace) -> int:
    """Handle the `rename` subcommand (module/attribute safe rename)."""
    # Apply router/profile for code tasks so any downstream LLM usage (future) is pinned
    try:
        from selfcoder.llm_router import apply_router_env as _route
        # Prefer first file ext or fall back to root
        f = (args.files[0] if getattr(args, 'files', None) else None) or getattr(args, 'root', None)
        # Ensure JSON mode is clean: suppress router verbose lines on stdout when --json is requested
        prev_verbose = os.getenv("NERION_ROUTER_VERBOSE")
        if getattr(args, "json", False):
            try:
                os.environ["NERION_ROUTER_VERBOSE"] = "0"
            except Exception:
                pass
        try:
            _route(instruction=f"rename {args.old} to {args.new}", file=str(f) if f else None, task='code')
        finally:
            if getattr(args, "json", False):
                # restore previous setting
                try:
                    if prev_verbose is None:
                        os.environ.pop("NERION_ROUTER_VERBOSE", None)
                    else:
                        os.environ["NERION_ROUTER_VERBOSE"] = prev_verbose
                except Exception:
                    pass
    except Exception:
        pass
    try:
        from selfcoder.policy.profile_resolver import decide as _dec, apply_env_scoped as _scope
        dec = _dec('apply_plan', signals={'has_rename': True})
        _scope(dec).__enter__()  # best-effort; process is short-lived
    except Exception:
        pass
    from selfcoder.actions.crossfile import (
        RenameSpec,
        apply_crossfile_rename,
        preview_crossfile_rename,
        _iter_pyfiles,
        update_import_paths,
    )

    if getattr(args, "imports_only", False):
        root_path = ensure_in_repo_auto(Path(args.root or Path.cwd()))
        if not getattr(args, "apply", False):
            preview_map = update_import_paths(root_path, args.old, args.new, include=args.include, exclude=args.exclude, preview=True)
            if getattr(args, "json", False):
                files_list = [{"path": k, "content": v} for k, v in preview_map.items()]
                print(json.dumps({"files": files_list}, indent=2))
            else:
                for p in preview_map.keys():
                    print(f"[DRYRUN] would edit: {p}")
            return 0
        changed = update_import_paths(root_path, args.old, args.new, include=args.include, exclude=args.exclude, preview=False)
        return 0 if changed else 1

    def build_argv():
        argv = ["--old", args.old, "--new", args.new]
        if args.attr_old:
            argv.extend(["--attr-old", args.attr_old])
        if args.attr_new:
            argv.extend(["--attr-new", args.attr_new])
        if args.root:
            argv.extend(["--root", str(args.root)])
        if args.include:
            for i in args.include:
                argv.extend(["--include", i])
        if args.exclude:
            for e in args.exclude:
                argv.extend(["--exclude", e])
        if args.apply:
            argv.append("--apply")
        argv.extend([str(f) for f in args.files])
        return argv

    if _maybe_simulate is not None:
        sim_rc = _maybe_simulate(args, "rename", build_argv)
        if sim_rc is not None:
            return sim_rc

    spec = RenameSpec(
        old_module=args.old,
        new_module=args.new,
        old_attr=getattr(args, "attr_old", None),
        new_attr=getattr(args, "attr_new", None),
    )

    files = [Path(f) for f in (args.files or [])]
    files = [ensure_in_repo_auto(p) for p in files]
    if not files and getattr(args, "root", None):
        root_path = ensure_in_repo_auto(Path(args.root))
        files = list(_iter_pyfiles(root_path, include=args.include, exclude=args.exclude))
    if not files:
        print("rename: no files found", file=sys.stderr)
        return 1

    if not getattr(args, "apply", False):
        would_change_map = preview_crossfile_rename([spec], files=files)
        if getattr(args, "json", False):
            # Normalize to a stable JSON shape: {"files": [{"path": ..., "content": ...}, ...]}
            structured: dict[str, object] = {}
            if isinstance(would_change_map, dict) and (
                ("files" in would_change_map) or ("changes" in would_change_map)
            ):
                structured = would_change_map
            else:
                files_list = []
                if isinstance(would_change_map, dict):
                    for path_str, content in would_change_map.items():
                        files_list.append({"path": str(path_str), "content": content})
                elif isinstance(would_change_map, list):
                    for item in would_change_map:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            files_list.append({"path": str(item[0]), "content": item[1]})
                structured = {"files": files_list}
            print(json.dumps(structured, indent=2))
        else:
            # Optional graph summary (non-JSON preview only)
            if getattr(args, "show_affected", False) and getattr(args, "attr_old", None):
                try:
                    from selfcoder.analysis.symbols import build_defs_uses
                    from selfcoder.analysis.symbols_graph import affected_files_for_symbol
                    root = Path(args.root) if getattr(args, "root", None) else Path.cwd()
                    idx = build_defs_uses(root, use_cache=True)
                    sym = str(getattr(args, "attr_old"))
                    defs = idx.get("defs", {}).get(sym) or []
                    uses = idx.get("uses", {}).get(sym) or []
                    aff = affected_files_for_symbol(sym, root)
                    print(f"[graph] defs for '{sym}': {len(defs)}; uses: {len(uses)}; transitive affected files: {len(aff)}")
                except Exception:
                    pass
            if not would_change_map:
                print("[rename] no changes would be made")
            else:
                for p in (
                    would_change_map.keys()
                    if isinstance(would_change_map, dict)
                    else would_change_map
                ):
                    print(f"[DRYRUN] would edit: {p}")
        return 0

    def _do_apply():
        _, changed = apply_crossfile_rename([spec], files=files)
        return bool(changed)

    def _do_check():
        res = healthcheck.run_all()
        return bool(res[0]) if isinstance(res, tuple) else bool(res)

    if _apply_with_rollback is None:
        # Fallback: apply without rollback if helper is unavailable
        ok = _do_apply() and _do_check()
        return 0 if ok else 1

    return 0 if _apply_with_rollback("pre-apply auto-rollback (rename)", _do_apply, _do_check) else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the `rename` subcommand on the given subparsers object."""
    # local simulate options so this module is self-contained
    sim = argparse.ArgumentParser(add_help=False)
    sim.add_argument("--simulate", action="store_true", help="Run in a shadow copy, test, and show diff")
    sim.add_argument("--simulate-json", action="store_true", help="Output simulation results as JSON")
    sim.add_argument("--simulate-keep", action="store_true", help="Do not delete the shadow directory")
    sim.add_argument("--simulate-dir", type=Path, help="Specify a directory for the shadow copy")
    sim.add_argument("--skip-pytest", action="store_true", help="Do not run pytest in simulation")
    sim.add_argument("--skip-healthcheck", action="store_true", help="Do not run healthcheck in simulation")
    sim.add_argument("--pytest-timeout", type=int, help="Timeout (seconds) for pytest during simulation")
    sim.add_argument("--healthcheck-timeout", type=int, help="Timeout (seconds) for healthcheck during simulation")

    p = subparsers.add_parser("rename", help="safely rename a module or update import statements", parents=[sim])
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--attr-old", dest="attr_old")
    p.add_argument("--attr-new", dest="attr_new")
    p.add_argument("files", nargs="*")
    p.add_argument("--root")
    p.add_argument("--include", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--json", action="store_true", help="in dry-run, emit machine-readable JSON preview")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--imports-only", action="store_true", help="Rename only import statements project-wide")
    p.add_argument("--show-affected", action="store_true", help="In preview, print graph-based defs/uses summary for the symbol (non-JSON)")
    p.set_defaults(func=cmd_rename)
