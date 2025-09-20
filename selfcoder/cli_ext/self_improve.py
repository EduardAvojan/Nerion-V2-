from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

# --- THE FIX: A custom JSON encoder that understands Path objects ---

class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj) # Convert Path objects to simple strings
        return super().default(obj)

# --- Helper: concise human summary for simulation vs. real apply ---
def _print_summary(res: dict) -> None:
    """Print a concise, human-readable summary for simulation vs. real apply."""
    simulated = bool(res.get("simulated"))
    applied = bool(res.get("applied"))
    rolled_back = bool(res.get("rolled_back"))
    writes_blocked_by = res.get("writes_blocked_by")
    staged = res.get("staged_artifacts") or []
    reason = res.get("reason")

    if simulated:
        print("[SAFE] Simulation mode is ON — repository writes are blocked")
        if writes_blocked_by:
            print(f"[SAFE] writes_blocked_by={writes_blocked_by}")
        print(f"[SUMMARY] SAFE simulation: 0 repo files changed ({len(staged)} staged artifacts in /tmp).")
        if staged:
            # show up to first 3 staged paths
            head = [str(p) for p in staged[:3]]
            print("[STAGED] " + ", ".join(head) + (" …" if len(staged) > 3 else ""))
        if reason:
            print(f"[CHECKS] result={reason}")
    else:
        if applied and not rolled_back:
            print("[WRITE] Applied plan successfully.")
        elif not applied and rolled_back:
            print("[ROLLBACK] Apply failed healthchecks/exception; restored snapshot.")
        else:
            print("[STATUS] Real apply finished with unexpected combination.")

def _cmd_scan(args: argparse.Namespace) -> int:
    try:
        from selfcoder.self_improve import scan as _scan
        report_path = _scan(paths=list(args.paths) if getattr(args, "paths", None) else None)
        print(str(report_path))
        return 0
    except Exception as exc:
        print(f"[self-improve scan] failed: {exc}")
        return 1


def _cmd_plan(args: argparse.Namespace) -> int:
    try:
        from selfcoder.self_improve import plan as _plan
        out_path = _plan(Path(args.report))
        print(str(out_path))
        return 0
    except Exception as exc:
        print(f"[self-improve plan] failed: {exc}")
        return 1


def _cmd_apply(args: argparse.Namespace) -> int:
    try:
        from selfcoder.self_improve import apply as _apply

        # Precedence: --apply (real) > --safe (force simulation) > default(sim)
        if getattr(args, "apply", False):
            is_simulation = False
        elif getattr(args, "safe", False):
            is_simulation = True
        else:
            is_simulation = True  # default to simulation

        res = _apply(Path(args.plan_file), simulate=is_simulation)

        # Human-readable banner/summary first
        _print_summary(res)

        # Then the machine-readable JSON
        print("\nJSON RESULT:")
        print(json.dumps(res, indent=2, cls=PathEncoder))

        if not res.get("applied", False) and not is_simulation:
            return 1
        return 0
    except Exception as exc:
        print(f"[self-improve apply] CRASHED with an unhandled exception: {exc}")
        traceback.print_exc()
        return 1


def register(subparsers) -> None:
    """Register the `self-improve` command group and subcommands."""
    psi = subparsers.add_parser("self-improve", help="proactive self-improvement pipeline")
    psi_sub = psi.add_subparsers(dest="self_improve_cmd", required=True)

    psi_scan = psi_sub.add_parser("scan", help="run analyzers and write a combined JSON report")
    psi_scan.add_argument("--paths", nargs="*", default=[], help="paths to include (default: repo roots)")
    psi_scan.set_defaults(func=_cmd_scan)

    psi_plan = psi_sub.add_parser("plan", help="convert a scan report to an AST plan JSON")
    psi_plan.add_argument("report", help="path to report JSON from 'self-improve scan'")
    psi_plan.set_defaults(func=_cmd_plan)

    psi_apply = psi_sub.add_parser("apply", help="apply an AST plan with safety checks (simulates by default)")
    psi_apply.add_argument("plan_file", help="path to plan JSON from 'self-improve plan'")
    psi_apply.add_argument("--safe", action="store_true", help="Force SAFE simulation (no repo writes). This is the default unless --apply is used.")
    psi_apply.add_argument("--apply", action="store_true", help="Apply the plan for real (overrides --safe; default without flags is SAFE simulation)")
    psi_apply.set_defaults(func=_cmd_apply)