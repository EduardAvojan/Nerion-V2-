"""Autotest command implementation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from selfcoder.vcs import git_ops
except Exception:
    class _GitOpsFallback:
        @staticmethod
        def snapshot(*_a, **_k):
            return "0"
        @staticmethod
        def restore_snapshot(*_a, **_k):
            return None
    git_ops = _GitOpsFallback()

try:
    from selfcoder.orchestrator import run_actions_on_file
except Exception:
    def run_actions_on_file(*_a, **_k):
        return False


def cmd_autotest(args: argparse.Namespace) -> int:
    """Generate and optionally run tests from a plan."""
    from selfcoder.planner.planner import plan_edits_from_nl
    from selfcoder import testgen
    if getattr(args, "plan_json", None):
        with open(args.plan_json, "r", encoding="utf-8") as fh:
            plan = json.load(fh)
    else:
        plan = plan_edits_from_nl(args.instruction, args.file)
    if not isinstance(plan, dict):
        print("[autotest] invalid plan object", file=sys.stderr)
        return 1
    target_file = plan.get("target_file") or args.file
    if not target_file:
        print("[autotest] missing target file", file=sys.stderr)
        return 2
    target_path = Path(target_file)
    ts_for_rollback = None
    if getattr(args, "apply", False) or getattr(args, "run", False):
        ts_for_rollback = git_ops.snapshot("pre-apply auto-rollback (autotest)")
        actions = plan.get("actions") or []
        if actions:
            try:
                changed = run_actions_on_file(target_path, actions, dry_run=False)
                print(f"[autotest] applied plan to {target_path} ({'changed' if changed else 'no change'})")
            except Exception as exc:
                print(f"[autotest] apply failed: {exc}", file=sys.stderr)
                git_ops.restore_snapshot(snapshot_ts=ts_for_rollback)
                return 1
    code = testgen.generate_tests_for_plan(plan, target_path)
    out_dir = Path("selfcoder/tests/generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"test_auto_{target_path.stem}.py"
    testgen.write_test_file(code, out_path)
    print(f"[autotest] wrote: {out_path}")
    rc = 0
    if getattr(args, "run", False):
        rc = testgen.run_pytest_on_paths([out_path])
        print(f"[autotest] pytest exit code: {rc}")
        if rc != 0 and ts_for_rollback:
            git_ops.restore_snapshot(snapshot_ts=ts_for_rollback)
    if getattr(args, "cov", False):
        from selfcoder import coverage_utils as covu
        cov_data = covu.run_pytest_with_coverage(pytest_args=[])
        cur_pct, delta = covu.compare_to_baseline(cov_data, covu.load_baseline())
        print(f"[coverage] total={cur_pct:.2f}% (Δ vs baseline {delta:+.2f}%)")
        if getattr(args, "fail_on_coverage_drop", False) and delta < 0:
            print("[coverage] FAIL: coverage dropped vs baseline")
            if rc == 0 and ts_for_rollback:
                git_ops.restore_snapshot(snapshot_ts=ts_for_rollback)
            rc = 1
        if getattr(args, "update_coverage_baseline", False):
            covu.save_baseline(cov_data)
            print("[coverage] baseline updated")
    return rc


def register_autotest(sub):
    """Register autotest subcommand."""
    sa = sub.add_parser("autotest", help="generate tests from a plan and optionally run them")
    sa.add_argument("--plan-json", help="path to plan JSON")
    sa.add_argument("-i", "--instruction", help="natural language instruction")
    sa.add_argument("-f", "--file", help="target file")
    sa.add_argument("--run", action="store_true", help="run pytest on the generated file")
    sa.add_argument("--apply", action="store_true", help="apply the generated plan before testing")
    sa.add_argument("--cov", action="store_true", help="Run coverage after generating tests")
    sa.add_argument("--fail-on-coverage-drop", action="store_true", help="Fail if total coverage drops vs baseline")
    sa.add_argument("--update-coverage-baseline", action="store_true", help="Update saved coverage baseline after run")
    sa.set_defaults(func=cmd_autotest)
