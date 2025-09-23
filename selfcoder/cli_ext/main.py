import argparse
import json
import sys
from pathlib import Path

from selfcoder.planner.planner import plan_edits_from_nl
from selfcoder.orchestrator import apply_plan
from selfcoder.plans.schema import validate_plan
from selfcoder.planner.apply_policy import apply_allowed, evaluate_apply_policy
from selfcoder.selfaudit import generate_improvement_plan
from selfcoder.scheduler import run_audit_job, should_enable_scheduler
from selfcoder.governor import evaluate as governor_evaluate
from selfcoder.governor import note_execution as governor_note_execution
from .self_learn import add_self_learn_subparser


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nerion-selfcoder", description="Self-coder CLI for planning and applying edits")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add_self_learn_subparser(sub)

    p_plan = sub.add_parser("plan", help="Generate a plan from natural language instruction")
    p_plan.add_argument("instruction", help="Natural language edit instruction")
    p_plan.add_argument("--file", help="Target file (optional)")
    p_plan.add_argument("--no-tests", action="store_true", help="Disable scaffolding of tests")

    p_apply = sub.add_parser("apply", help="Apply a plan from JSON file")
    p_apply.add_argument("planfile", help="Path to JSON plan")
    p_apply.add_argument("--preview", action="store_true", help="Preview a unified diff for the plan without applying")
    p_apply.add_argument("--heal", help="Comma-separated list of healers to run (e.g., format,isort)")
    p_apply.add_argument("--force-apply", action="store_true", help="Override apply policy gating and proceed anyway")
    p_apply.add_argument("--force-governor", action="store_true", help="Bypass governor scheduling / rate limits")

    p_audit = sub.add_parser("audit", help="Run self-audit to generate an improvement plan")
    p_audit.add_argument("--root", default=".", help="Project root to audit")
    p_audit.add_argument("--plan-out", help="Optional path to write the generated plan JSON")
    p_audit.add_argument("--preview", action="store_true", help="Preview a unified diff for the generated plan without applying")
    p_audit.add_argument("--apply", action="store_true", help="Apply the generated plan directly")
    p_audit.add_argument("--heal", help="Comma-separated list of healers to run when applying")
    p_audit.add_argument("--force-apply", action="store_true", help="Override apply policy gating when --apply is set")
    p_audit.add_argument("--force-governor", action="store_true", help="Bypass governor scheduling / rate limits when applying")

    p_asched = sub.add_parser("audit-schedule", help="Run periodic self-audit (controlled by env SELFAUDIT_ENABLE/INTERVAL)")
    p_asched.add_argument("--root", default=".", help="Project root to audit")
    p_asched.add_argument("--once", action="store_true", help="Run once then exit (ignores interval)")

    args = parser.parse_args(argv)

    if args.cmd == "plan":
        plan = plan_edits_from_nl(args.instruction, file=args.file, scaffold_tests=not args.no_tests)
        # Validate for safety, but emit the original dict so it remains JSON-serializable
        validate_plan(plan)
        print(json.dumps(plan, indent=2))
        return 0

    if args.cmd == "apply":
        path = Path(args.planfile)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERR] Could not read plan: {e}", file=sys.stderr)
            return 1
        try:
            # Validate for safety, but pass the original dict to the orchestrator
            validate_plan(data)
        except Exception as e:
            print(f"[ERR] Invalid plan: {e}", file=sys.stderr)
            return 1
        healers = args.heal.split(",") if args.heal else None

        governor_decision = None
        if not args.preview:
            decision = evaluate_apply_policy(data)
            force_apply = getattr(args, "force_apply", False)
            if not apply_allowed(decision, force=force_apply):
                header = "BLOCK" if decision.is_blocked() else "REVIEW"
                print(f"[policy] {header}: apply requires manual approval (policy={decision.policy})")
                for reason in decision.reasons:
                    print(f"[policy] - {reason}")
                print("[policy] re-run with --force-apply to override.")
                return 3 if decision.is_blocked() else 2
            if force_apply:
                if decision.is_blocked():
                    print("[policy] forcing apply despite block decision; proceed with caution")
                elif decision.requires_manual_review():
                    print("[policy] forcing apply despite review gate")
            else:
                print(f"[policy] apply decision: {decision.decision} (policy={decision.policy})")
                for reason in decision.reasons:
                    print(f"[policy] - {reason}")

            try:
                governor_decision = governor_evaluate(
                    "cli_ext.apply",
                    override=(force_apply or getattr(args, "force_governor", False)),
                )
            except Exception:
                governor_decision = None

            if governor_decision and governor_decision.is_blocked():
                print(f"[governor] BLOCK: {governor_decision.code}")
                for reason in governor_decision.reasons:
                    print(f"[governor] - {reason}")
                if governor_decision.next_allowed_local:
                    print(f"[governor] next allowed at {governor_decision.next_allowed_local}")
                print('[governor] re-run with --force-governor or --force-apply to override.')
                return 4

            if governor_decision and governor_decision.override_used:
                print("[governor] override flag detected; proceeding under manual override")

            governor_note_execution("cli_ext.apply")

        touched = apply_plan(data, preview=args.preview, healers=healers)
        print("Touched files:")
        for f in touched:
            print(" -", f)
        return 0

    if args.cmd == "audit":
        root = Path(args.root)
        plan = generate_improvement_plan(root)
        validate_plan(plan)
        if args.preview:
            apply_plan(plan, preview=True)
            return 0
        if args.apply:
            decision = evaluate_apply_policy(plan)
            force_apply = getattr(args, "force_apply", False)
            if not apply_allowed(decision, force=force_apply):
                header = "BLOCK" if decision.is_blocked() else "REVIEW"
                print(f"[policy] {header}: apply requires manual approval (policy={decision.policy})")
                for reason in decision.reasons:
                    print(f"[policy] - {reason}")
                print("[policy] re-run with --force-apply to override.")
                return 3 if decision.is_blocked() else 2
            if force_apply:
                if decision.is_blocked():
                    print("[policy] forcing apply despite block decision; proceed with caution")
                elif decision.requires_manual_review():
                    print("[policy] forcing apply despite review gate")
            else:
                print(f"[policy] apply decision: {decision.decision} (policy={decision.policy})")
                for reason in decision.reasons:
                    print(f"[policy] - {reason}")
            healers = args.heal.split(",") if args.heal else None
            try:
                governor_decision = governor_evaluate(
                    "cli_ext.audit_apply",
                    override=(force_apply or getattr(args, "force_governor", False)),
                )
            except Exception:
                governor_decision = None

            if governor_decision and governor_decision.is_blocked():
                print(f"[governor] BLOCK: {governor_decision.code}")
                for reason in governor_decision.reasons:
                    print(f"[governor] - {reason}")
                if governor_decision.next_allowed_local:
                    print(f"[governor] next allowed at {governor_decision.next_allowed_local}")
                print('[governor] re-run with --force-governor or --force-apply to override.')
                return 4

            if governor_decision and governor_decision.override_used:
                print("[governor] override flag detected; proceeding under manual override")

            governor_note_execution("cli_ext.audit_apply")
            apply_plan(plan, healers=healers)
            return 0
        if args.plan_out:
            Path(args.plan_out).write_text(json.dumps(plan, indent=2), encoding="utf-8")
        else:
            print(json.dumps(plan, indent=2))
        return 0

    if args.cmd == "audit-schedule":
        root = Path(args.root)
        interval = should_enable_scheduler()
        if interval is None:
            print("[scheduler] disabled by env SELFAUDIT_ENABLE")
            return 1
        run_audit_job(root, interval, once=args.once)
        return 0

    handler = getattr(args, "_handler", None)
    if handler:
        return int(handler(args))

    return 1


if __name__ == "__main__":
    sys.exit(main())
