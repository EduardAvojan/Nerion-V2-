from __future__ import annotations
import argparse
import json

from selfcoder.analysis import deps as deps_mod


def _parse_csv(val: str | None) -> set[str]:
    if not val:
        return set()
    return {p.strip().lower() for p in val.split(',') if p.strip()}


def cmd_scan(args: argparse.Namespace) -> int:
    report = deps_mod.scan(offline=bool(getattr(args, "offline", False)))
    try:
        artifact = deps_mod.persist_scan(report)
        # include path without changing the expected top-level keys
        report["artifact_path"] = str(artifact)
    except Exception:
        pass
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    scan = deps_mod.scan(offline=bool(getattr(args, "offline", False)))
    only = _parse_csv(getattr(args, "only", None))
    exclude = _parse_csv(getattr(args, "exclude", None))
    plan = deps_mod.make_upgrade_plan(
        scan.get("outdated", []),
        policy=args.policy,
        only=only,
        exclude=exclude,
    )
    try:
        artifact = deps_mod.persist_plan(plan)
        plan["artifact_path"] = str(artifact)
    except Exception:
        pass
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    scan = deps_mod.scan(offline=bool(getattr(args, "offline", False)))
    only = _parse_csv(getattr(args, "only", None))
    exclude = _parse_csv(getattr(args, "exclude", None))
    plan = deps_mod.make_upgrade_plan(
        scan.get("outdated", []),
        policy=args.policy,
        only=only,
        exclude=exclude,
    )
    result = deps_mod.apply_plan(plan, dry_run=bool(args.dry_run))
    bundle = {"plan": plan, "result": result}
    try:
        artifact = deps_mod.persist_apply(bundle)
        bundle["artifact_path"] = str(artifact)
    except Exception:
        pass
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return 0 if result.get("applied") or result.get("dry_run") else 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("deps", help="Dependency maintenance (scan/plan/apply)")
    sp = p.add_subparsers(dest="deps_cmd")

    p_scan = sp.add_parser("scan", help="Scan environment: freeze, outdated, audit")
    p_scan.add_argument("--offline", action="store_true", help="Use offline providers (no subprocess calls) for fast, deterministic scans.")
    p_scan.set_defaults(func=cmd_scan)

    p_plan = sp.add_parser("plan", help="Plan upgrades based on scan")
    p_plan.add_argument("--offline", action="store_true", help="Use offline providers (no subprocess calls) for fast, deterministic scans.")
    p_plan.add_argument("--policy", choices=["patch", "minor", "major"], default="patch")
    p_plan.add_argument("--only", metavar="CSV", help="Comma-separated package names to include (case-insensitive)")
    p_plan.add_argument("--exclude", metavar="CSV", help="Comma-separated package names to exclude (case-insensitive)")
    p_plan.set_defaults(func=cmd_plan)

    p_apply = sp.add_parser("apply", help="Apply upgrade plan (defaults to dry-run)")
    p_apply.add_argument("--offline", action="store_true", help="Use offline providers (no subprocess calls) for fast, deterministic scans.")
    p_apply.add_argument("--policy", choices=["patch", "minor", "major"], default="patch")
    p_apply.add_argument("--dry-run", action="store_true", help="Do not execute pip, only show commands")
    p_apply.add_argument("--only", metavar="CSV", help="Comma-separated package names to include (case-insensitive)")
    p_apply.add_argument("--exclude", metavar="CSV", help="Comma-separated package names to exclude (case-insensitive)")
    p_apply.set_defaults(func=cmd_apply)