
import argparse
import sys
import json
from pathlib import Path

from selfcoder.orchestrator import run_actions_on_file
from selfcoder import healthcheck


def _maybe_simulate_wrapper(args, argv_builder):
    # Lazy import to avoid circulars at module import time
    from selfcoder.cli import _maybe_simulate
    return _maybe_simulate(args, "batch", argv_builder)


def _apply_with_rollback_wrapper(snapshot_message, apply_fn, check_fn=None):
    # Lazy import to avoid circulars
    from selfcoder.cli import _apply_with_rollback
    return _apply_with_rollback(snapshot_message, apply_fn, check_fn)


def cmd_batch(args: argparse.Namespace) -> int:
    # Route coder by first file; scope a safe profile for apply
    try:
        from selfcoder.llm_router import apply_router_env as _route
        f = args.files[0] if getattr(args, 'files', None) else None
        _route(instruction=None, file=str(f) if f else None, task='code')
    except Exception:
        pass
    try:
        from selfcoder.policy.profile_resolver import decide as _dec, apply_env_scoped as _scope
        dec = _dec('apply_plan', signals={'files_count': len(getattr(args, 'files', []) or [])})
        _scope(dec).__enter__()
    except Exception:
        pass
    def build_argv():
        argv = ["--actions-file", str(args.actions_file)] + [str(f) for f in args.files]
        if args.apply:
            argv.append("--apply")
        return argv

    sim_rc = _maybe_simulate_wrapper(args, build_argv)
    if sim_rc is not None:
        return sim_rc

    try:
        with open(args.actions_file, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        actions = blob.get("actions") if isinstance(blob, dict) else blob if isinstance(blob, list) else None
        if actions is None:
            print("error: actions file must be a list or object with 'actions'", flush=True)
            return 2
        files = [Path(f) for f in args.files]
        if args.dry_run:
            changed = sum(1 for fp in files if run_actions_on_file(fp, actions, dry_run=True))
            print(f"[batch] dry-run: processed {len(files)} file(s); {changed} would change")
            return 0
        else:
            changed_count = 0

            def _do_apply():
                nonlocal changed_count
                for fp in files:
                    if run_actions_on_file(fp, actions, dry_run=False):
                        changed_count += 1
                return True

            def _do_check():
                res = healthcheck.run_all()
                return bool(res[0]) if isinstance(res, tuple) else bool(res)

            ok = _apply_with_rollback_wrapper("pre-apply auto-rollback (batch)", _do_apply, _do_check)
            if not ok:
                print("[batch] healthcheck failed; rolled back", file=sys.stderr)
                return 1
            print(f"[batch] processed {len(files)} file(s); {changed_count} changed")
            return 0
    except Exception as exc:
        print(f"[batch] failed: {exc}", file=sys.stderr)
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the `batch` subcommand with the given top-level subparsers."""
    import argparse as _ap

    sim_parser = _ap.ArgumentParser(add_help=False)
    sim_parser.add_argument("--simulate", action="store_true", help="Run in a shadow copy, test, and show diff")
    sim_parser.add_argument("--simulate-json", action="store_true", help="Output simulation results as JSON")
    sim_parser.add_argument("--simulate-keep", action="store_true", help="Do not delete the shadow directory")
    sim_parser.add_argument("--simulate-dir", type=Path, help="Specify a directory for the shadow copy")
    sim_parser.add_argument("--skip-pytest", action="store_true", help="Do not run pytest in simulation")
    sim_parser.add_argument("--skip-healthcheck", action="store_true", help="Do not run healthcheck in simulation")
    sim_parser.add_argument("--pytest-timeout", type=int, help="Timeout (seconds) for pytest during simulation")
    sim_parser.add_argument("--healthcheck-timeout", type=int, help="Timeout (seconds) for healthcheck during simulation")

    sb = subparsers.add_parser("batch", help="apply actions to multiple files", parents=[sim_parser])
    sb.add_argument("--actions-file", required=True)
    sb.add_argument("files", nargs="+")
    sb.add_argument("--dry-run", action="store_true")
    sb.set_defaults(func=cmd_batch)
