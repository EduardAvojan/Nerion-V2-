from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
try:
    from ops.security.safe_subprocess import safe_run
except Exception:
    import subprocess
    def safe_run(argv, **kwargs):
        return subprocess.run(argv, **{k: v for k, v in kwargs.items() if k in ("cwd","timeout","check","capture_output")})
import sys
from pathlib import Path
from ops.telemetry.logger import log
from datetime import datetime, timezone
from selfcoder.scoring import score_plan
from selfcoder.artifacts import PlanArtifact, SimResult, save_artifact
# Import from cli module for test compatibility (tests may monkeypatch this)
import selfcoder.cli as cli_module
from selfcoder.planner.apply_policy import (
    apply_allowed,
    evaluate_apply_policy,
)
from selfcoder.planner.utils import attach_brief_metadata
from selfcoder.governor import evaluate as governor_evaluate
from selfcoder.governor import note_execution as governor_note_execution

from selfcoder import healthcheck
from selfcoder.orchestrator import run_actions_on_file

from selfcoder.vcs import git_ops


def _apply_with_rollback(snapshot_message: str, apply_fn, check_fn=None) -> bool:
    """
    Take a VCS snapshot, call apply_fn(), then optionally call check_fn().
    If apply_fn raises or check_fn returns False, restore the snapshot and return False.
    Returns True on success.
    """
    ts_raw = git_ops.snapshot(snapshot_message)
    # Normalize snapshot token to a string for restore_snapshot
    # In SAFE mode, snapshot returns a list of files - we can't restore from this
    ts_is_safe_mode = isinstance(ts_raw, (list, tuple))
    if ts_is_safe_mode:
        ts = ts_raw[0] if ts_raw else ""
    elif isinstance(ts_raw, dict):
        ts = ts_raw.get("ts") or ts_raw.get("timestamp") or next(iter(ts_raw.values()), None)
    else:
        ts = ts_raw
    ts = str(ts)

    try:
        _ok_apply = bool(apply_fn())
    except Exception as exc:
        print(f"[apply] failed during apply: {exc}", file=sys.stderr)
        if not ts_is_safe_mode and ts:
            git_ops.restore_snapshot(snapshot_ts=ts)
        return False

    ok = True
    if check_fn is not None:
        try:
            ok = bool(check_fn())
        except Exception as exc:
            ok = False
            print(f"[check] failed to run: {exc}", file=sys.stderr)

    if not ok and not ts_is_safe_mode and ts:
        git_ops.restore_snapshot(snapshot_ts=ts)
    return ok


def _safe_rmtree(path: Path) -> None:
    try:
        tmp = Path(tempfile.gettempdir()).resolve()
        p = Path(path).resolve()
        if p.is_dir() and (tmp in p.parents or p == tmp) and len(p.parts) > 3:
            shutil.rmtree(p, ignore_errors=True)
            log("SIM", "Removed shadow directory", {"shadow": p.as_posix()})
        else:
            print(f"[simulate] Refusing to delete suspicious path: {p}")
    except Exception as e:
        print(f"[simulate] Failed to remove shadow dir: {e}")


def _maybe_simulate(args, cmd_name, argv_builder):
    if not getattr(args, "simulate", False):
        return None  # No simulation, caller continues with the actual apply

    from selfcoder.simulation import (
        make_shadow_copy,
        run_tests_and_healthcheck,
        compute_diff,
        _rewrite_paths_for_shadow,
    )

    root = Path.cwd()
    shadow = make_shadow_copy(root, getattr(args, "simulate_dir", None))

    original_argv = argv_builder()
    # Rewrite paths to point inside the shadow copy
    shadow_argv = _rewrite_paths_for_shadow(original_argv, root, shadow)

    # Run the command in the shadow repository
    full_command = [sys.executable, "-m", "selfcoder.cli", cmd_name] + shadow_argv
    result = safe_run(full_command, cwd=shadow, timeout=300, check=False, capture_output=False)
    rc_cmd = result.returncode

    skip_pytest = getattr(args, "skip_pytest", False)
    skip_healthcheck = getattr(args, "skip_healthcheck", False)
    pytest_timeout = getattr(args, "pytest_timeout", None)
    healthcheck_timeout = getattr(args, "healthcheck_timeout", None)
    results = run_tests_and_healthcheck(
        shadow,
        skip_pytest=skip_pytest,
        skip_healthcheck=skip_healthcheck,
        pytest_timeout=pytest_timeout,
        healthcheck_timeout=healthcheck_timeout,
    )
    diff = compute_diff(root, shadow)
    changed_files = diff.get("files", [])
    changed = bool(changed_files)

    # Score the plan using simulation signals and save an artifact
    try:
        pytest_entry = results.get("pytest", {})
        health_entry = results.get("healthcheck", {})
        pytest_rc = None if pytest_entry.get("skipped") else pytest_entry.get("rc")
        health_ok = health_entry.get("ok") if not health_entry.get("skipped") else True
        files_touched = [str(x) for x in changed_files]
        diff_size = len(diff.get("text", "")) if isinstance(diff.get("text"), str) else None

        # Rebuild the plan dict for scoring context
        from selfcoder.planner.planner import plan_edits_from_nl as _plan_from_nl
        sim_plan = _plan_from_nl(args.instruction, args.file, scaffold_tests=getattr(args, "scaffold_tests", True))
        sim_summary = {
            "pytest_rc": pytest_rc,
            "health_ok": health_ok,
            "files_touched": files_touched,
            "diff_size": diff_size,
            "checks": results,
        }
        score, why = score_plan(sim_plan, sim_summary)
        print(f"[SCORE] {score} ({why})")
        artifact = PlanArtifact(
            ts=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
            origin="cli.plan.simulate",
            score=score,
            rationale=why,
            plan=sim_plan,
            files_touched=files_touched,
            sim=SimResult(
                pytest_rc=pytest_rc,
                health_ok=health_ok,
                pytest_out=pytest_entry.get("stdout"),
                health_out=health_entry.get("stdout"),
                checks=results,
            ),
            meta={"shadow_cmd_rc": rc_cmd},
        )
        save_artifact(artifact)
    except Exception:
        pass

    # Present the simulation results
    print("\n--- Simulation Report ---")
    print(f"[simulate] Shadow Directory: {shadow}")
    print(f"[simulate] Command exit code: {rc_cmd}")
    for name, entry in results.items():
        label = name.replace("_", " ").title()
        if name == "pytest":
            label = "Pytest"
        elif name == "healthcheck":
            label = "Healthcheck"
        if entry.get("skipped"):
            reason = entry.get("reason") or ""
            print(f"[simulate] {label} skipped: {reason}")
        else:
            print(f"[simulate] {label} exit code: {entry.get('rc')}")
    print(f"--- Diff ---\n{diff['text']}")

    out = None
    if getattr(args, "simulate_json", False):
        out = {
            "shadow_dir": str(shadow),
            "cmd_exit": rc_cmd,
            "changed": changed,
            "changed_files": changed_files,
            "diff_text": diff["text"],
            "checks": results,
        }

    if not getattr(args, "simulate_keep", False):
        print("[simulate] Cleaning up shadow directory...")
        _safe_rmtree(shadow)

    if out is not None:
        print(json.dumps(out, indent=2))

    failed_checks = [
        name
        for name, entry in results.items()
        if not entry.get("skipped") and entry.get("rc") not in (0, None)
    ]
    failed = rc_cmd != 0 or bool(failed_checks)
    return 1 if failed else 0


# ---- command implementation ------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    # Route LLM/coder settings early (opt-in with --llm)
    try:
        if bool(getattr(args, "llm", False)):
            from selfcoder.llm_router import apply_router_env as _route
            _route(instruction=getattr(args, "instruction", None), file=getattr(args, "file", None), task="code")
    except Exception:
        pass
    def build_argv():
        argv = ["-i", args.instruction]
        if args.file:
            argv.extend(["-f", str(args.file)])
        if args.apply:
            argv.append("--apply")
        # propagate scaffold_tests flag
        if getattr(args, "scaffold_tests", True) is False:
            argv.append("--no-scaffold-tests")
        else:
            argv.append("--scaffold-tests")
        # propagate LLM/grammar flags into the shadow run when simulating
        if getattr(args, "llm", False):
            argv.append("--llm")
        if getattr(args, "json_grammar", False):
            argv.append("--json-grammar")
        return argv

    sim_rc = _maybe_simulate(args, "plan", build_argv)
    if sim_rc is not None:
        return sim_rc

    # Build plan (LLM or heuristic), then sanitize and cache
    plan: dict
    brief_context = None
    try:
        brief_context = cli_module.build_planner_context(
            args.instruction,
            target_file=str(args.file) if args.file else None,
        )
    except Exception:
        brief_context = None
    use_llm = bool(getattr(args, "llm", False))
    if use_llm:
        # User-friendly prep and strict knobs for LLM mode
        try:
            from selfcoder.orchestrator import prepare_for_prompt as _prep
            _prep(getattr(args, "instruction", ""))
        except Exception:
            pass
        # Default to strict behavior unless user has explicitly opted out
        os.environ.setdefault("NERION_LLM_STRICT", "1")
        if getattr(args, "json_grammar", False):
            os.environ.setdefault("NERION_JSON_GRAMMAR", "1")
        try:
            from selfcoder.planner.llm_planner import plan_with_llm as _plan_llm
            plan = _plan_llm(args.instruction, args.file, brief_context=brief_context)
        except Exception as e:
            # If strict mode is active, surface the failure and exit non-zero
            if os.getenv("NERION_LLM_STRICT"):
                print(f"[plan] LLM planner failed: {e}", file=sys.stderr)
                return 2
            from selfcoder.planner.planner import plan_edits_from_nl as _nl
            plan = _nl(
                args.instruction,
                args.file,
                scaffold_tests=getattr(args, "scaffold_tests", True),
                brief_context=brief_context,
            )
    else:
        from selfcoder.planner.planner import plan_edits_from_nl
        plan = plan_edits_from_nl(
            args.instruction,
            args.file,
            scaffold_tests=getattr(args, "scaffold_tests", True),
            brief_context=brief_context,
        )

    # Sanitize and cache plan (keyed by repo fingerprint, target, instruction)
    try:
        from selfcoder.planner.utils import sanitize_plan, repo_fingerprint, load_plan_cache, save_plan_cache
        root = Path.cwd()
        fp = repo_fingerprint(root)
        target = str(args.file or (plan.get("target_file") or ""))
        brief_id = ""
        if brief_context and isinstance(brief_context, dict):
            brief_id = str((brief_context.get("brief") or {}).get("id") or "")
            decision = brief_context.get("decision") or ""
            if decision:
                brief_id = f"{brief_id}:{decision}"
        key = f"{fp}:{target}:{args.instruction.strip()}:{brief_id}"
        cache_path = root / ".nerion" / "plan_cache.json"
        cache = load_plan_cache(cache_path)
        if key in cache:
            plan = cache[key]
        else:
            try:
                plan = sanitize_plan(plan)
            except Exception as e:
                # In strict mode, propagate sanitization failures
                if os.getenv("NERION_LLM_STRICT"):
                    print(f"[plan] sanitize failed: {e}", file=sys.stderr)
                    return 2
            cache[key] = plan
            try:
                save_plan_cache(cache_path, cache)
            except Exception:
                pass
    except Exception:
        # best-effort; keep original plan
        pass

    plan = attach_brief_metadata(plan, brief_context)

    # Print compact JSON first so tests/parsers can read the first line
    print(json.dumps(plan, separators=(",", ":"), ensure_ascii=False))
    try:
        score, why = score_plan(plan, None)
        print(f"[SCORE] {score} ({why})")
        artifact = PlanArtifact(
            ts=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
            origin="cli.plan",
            score=score,
            rationale=why,
            plan=plan,
            files_touched=[str(args.file)] if args.file else [],
            sim=None,
            meta={}
        )
        save_artifact(artifact)
    except Exception:
        pass
    if getattr(args, "apply", False):
        actions = plan.get("actions") or []
        target = plan.get("target_file") or args.file
        if not target:
            print("[apply] no target_file provided; nothing applied", file=sys.stderr)
            return 1

        force_apply = getattr(args, "force_apply", False)
        governor_override = force_apply or getattr(args, "force_governor", False)
        decision = evaluate_apply_policy(plan)
        try:
            log(
                "APPLY_POLICY",
                "cli.plan",
                {
                    "decision": decision.decision,
                    "policy": decision.policy,
                    "reasons": list(decision.reasons),
                },
            )
        except Exception:
            pass

        if not apply_allowed(decision, force=force_apply):
            header = "BLOCK" if decision.is_blocked() else "REVIEW"
            print(f"[policy] {header}: apply requires manual approval (policy={decision.policy})")
            for reason in decision.reasons:
                print(f"[policy] - {reason}")
            print('[policy] re-run with --force-apply to override.')
            return 3 if decision.is_blocked() else 2

        if force_apply and decision.is_blocked():
            print("[policy] forcing apply despite block decision; proceed with caution")
        elif force_apply and decision.requires_manual_review():
            print("[policy] forcing apply despite review gate")
        else:
            print(f"[policy] apply decision: {decision.decision} (policy={decision.policy})")
            for reason in decision.reasons:
                print(f"[policy] - {reason}")

        governor_decision = None
        try:
            governor_decision = governor_evaluate(
                "cli.plan.apply",
                override=governor_override,
            )
        except Exception:
            governor_decision = None

        if governor_decision and governor_decision.is_blocked():
            try:
                log(
                    "GOVERNOR",
                    "cli.plan",
                    {
                        "decision": governor_decision.code,
                        "reasons": list(governor_decision.reasons),
                        "next_allowed": governor_decision.next_allowed_utc,
                    },
                )
            except Exception:
                pass
            print(f"[governor] BLOCK: {governor_decision.code}")
            for reason in governor_decision.reasons:
                print(f"[governor] - {reason}")
            if governor_decision.next_allowed_local:
                print(f"[governor] next allowed at {governor_decision.next_allowed_local}")
            print('[governor] re-run with --force-governor or --force-apply to override.')
            return 4

        if governor_decision and governor_decision.override_used:
            print("[governor] override flag detected; proceeding under manual override")

        def _do_apply() -> bool:
            changed = run_actions_on_file(Path(target), actions, dry_run=False)
            print(f"[apply] {'wrote' if changed else 'no change needed'}: {target}")
            return True

        def _do_check() -> bool:
            res = healthcheck.run_all()
            health_ok = bool(res[0]) if isinstance(res, tuple) else bool(res)
            print(f"[apply] post-check healthcheck: {'OK' if health_ok else 'FAIL'}")

            from selfcoder.verifier import failed_checks, run_post_apply_checks

            verify_results = run_post_apply_checks(Path.cwd())
            try:
                log(
                    "VERIFY",
                    "cli.plan",
                    {
                        "health_ok": health_ok,
                        "checks": {
                            name: {
                                "rc": entry.get("rc"),
                                "skipped": entry.get("skipped"),
                                "reason": entry.get("reason"),
                                "duration": entry.get("duration"),
                            }
                            for name, entry in verify_results.items()
                        },
                    },
                )
            except Exception:
                pass

            for name, entry in verify_results.items():
                label = name.replace("_", " ").title()
                if entry.get("skipped"):
                    reason = entry.get("reason") or ""
                    print(f"[verify] {label} skipped: {reason}")
                else:
                    print(f"[verify] {label} exit code: {entry.get('rc')}")

            failed = failed_checks(verify_results)
            return health_ok and not failed

        governor_note_execution("cli.plan.apply")

        rc = _apply_with_rollback("pre-apply auto-rollback (plan)", _do_apply, _do_check)
        if governor_decision:
            try:
                log(
                    "GOVERNOR",
                    "cli.plan",
                    {
                        "decision": governor_decision.code,
                        "override": governor_decision.override_used,
                    },
                )
            except Exception:
                pass
        return 0 if rc else 1
    return 0


def register(subparsers) -> None:
    """Register the `plan` subcommand on the provided subparsers."""
    sim_parser = argparse.ArgumentParser(add_help=False)
    sim_parser.add_argument("--simulate", action="store_true", help="Run in a shadow copy, test, and show diff")
    sim_parser.add_argument("--simulate-json", action="store_true", help="Output simulation results as JSON")
    sim_parser.add_argument("--simulate-keep", action="store_true", help="Do not delete the shadow directory")
    sim_parser.add_argument("--simulate-dir", type=Path, help="Specify a directory for the shadow copy")
    sim_parser.add_argument("--skip-pytest", action="store_true", help="Do not run pytest in simulation")
    sim_parser.add_argument("--skip-healthcheck", action="store_true", help="Do not run healthcheck in simulation")
    sim_parser.add_argument("--pytest-timeout", type=int, help="Timeout (seconds) for pytest during simulation")
    sim_parser.add_argument("--healthcheck-timeout", type=int, help="Timeout (seconds) for healthcheck during simulation")

    sp_plan = subparsers.add_parser("plan", help="plan edits from natural language", parents=[sim_parser])
    sp_plan.add_argument("--apply", action="store_true", help="execute the plan")
    sp_plan.add_argument("-i", "--instruction", required=True)
    sp_plan.add_argument("-f", "--file", help="Optional file path")
    sp_plan.add_argument(
        "--scaffold-tests",
        dest="scaffold_tests",
        action="store_true",
        default=True,
        help="auto-create/refresh pytest scaffolds for created/modified symbols (default: on)",
    )
    sp_plan.add_argument(
        "--no-scaffold-tests",
        dest="scaffold_tests",
        action="store_false",
        help="disable automatic test scaffolding",
    )
    # Phase 2: Grammar-constrained planning knobs
    sp_plan.add_argument("--llm", action="store_true", help="Use LLM (Coder V2) planner instead of heuristics")
    sp_plan.add_argument("--json-grammar", dest="json_grammar", action="store_true", help="Force JSON-mode decoding for LLM planner (strict)")
    sp_plan.add_argument(
        "--force-apply",
        action="store_true",
        help="Override apply policy gating (manual operator approval)",
    )
    sp_plan.add_argument(
        "--force-governor",
        action="store_true",
        help="Bypass governor scheduling / rate limits for this apply",
    )
    sp_plan.set_defaults(func=cmd_plan)
