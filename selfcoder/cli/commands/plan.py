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

from selfcoder import healthcheck
from selfcoder.orchestrator import run_actions_on_file

from selfcoder.vcs import git_ops


def _apply_with_rollback(snapshot_message: str, apply_fn, check_fn=None) -> bool:
    """
    Take a VCS snapshot, call apply_fn(), then optionally call check_fn().
    If apply_fn raises or check_fn returns False, restore the snapshot and return False.
    Returns True on success.
    """
    ts = git_ops.snapshot(snapshot_message)
    try:
        _ok_apply = bool(apply_fn())
    except Exception as exc:
        print(f"[apply] failed during apply: {exc}", file=sys.stderr)
        git_ops.restore_snapshot(snapshot_ts=ts)
        return False

    ok = True
    if check_fn is not None:
        try:
            ok = bool(check_fn())
        except Exception as exc:
            ok = False
            print(f"[check] failed to run: {exc}", file=sys.stderr)

    if not ok:
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
        pytest_rc = results["pytest"]["rc"] if not skip_pytest else None
        health_rc = results["healthcheck"]["rc"] if not skip_healthcheck else None
        health_ok = (health_rc == 0) if (health_rc is not None) else None
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
            sim=SimResult(pytest_rc=pytest_rc, health_ok=health_ok),
            meta={"shadow_cmd_rc": rc_cmd},
        )
        save_artifact(artifact)
    except Exception:
        pass

    # Present the simulation results
    print("\n--- Simulation Report ---")
    print(f"[simulate] Shadow Directory: {shadow}")
    print(f"[simulate] Command exit code: {rc_cmd}")
    print(f"[simulate] Pytest exit code: {results['pytest']['rc']}")
    print(f"[simulate] Healthcheck exit code: {results['healthcheck']['rc']}")
    print(f"--- Diff ---\n{diff['text']}")

    out = None
    if getattr(args, "simulate_json", False):
        out = {
            "shadow_dir": str(shadow),
            "cmd_exit": rc_cmd,
            "pytest_exit": results["pytest"]["rc"] if not skip_pytest else None,
            "healthcheck_exit": results["healthcheck"]["rc"] if not skip_healthcheck else None,
            "changed": changed,
            "changed_files": changed_files,
            "diff_text": diff["text"],
        }

    if not getattr(args, "simulate_keep", False):
        print("[simulate] Cleaning up shadow directory...")
        _safe_rmtree(shadow)

    if out is not None:
        print(json.dumps(out, indent=2))

    failed = rc_cmd != 0 or (results["pytest"]["rc"] != 0 if not skip_pytest else False) or (results["healthcheck"]["rc"] != 0 if not skip_healthcheck else False)
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
            plan = _plan_llm(args.instruction, args.file)
        except Exception as e:
            # If strict mode is active, surface the failure and exit non-zero
            if os.getenv("NERION_LLM_STRICT"):
                print(f"[plan] LLM planner failed: {e}", file=sys.stderr)
                return 2
            from selfcoder.planner.planner import plan_edits_from_nl as _nl
            plan = _nl(args.instruction, args.file, scaffold_tests=getattr(args, "scaffold_tests", True))
    else:
        from selfcoder.planner.planner import plan_edits_from_nl
        plan = plan_edits_from_nl(args.instruction, args.file, scaffold_tests=getattr(args, "scaffold_tests", True))

    # Sanitize and cache plan (keyed by repo fingerprint, target, instruction)
    try:
        from selfcoder.planner.utils import sanitize_plan, repo_fingerprint, load_plan_cache, save_plan_cache
        root = Path.cwd()
        fp = repo_fingerprint(root)
        target = str(args.file or (plan.get("target_file") or ""))
        key = f"{fp}:{target}:{args.instruction.strip()}"
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

        def _do_apply() -> bool:
            changed = run_actions_on_file(Path(target), actions, dry_run=False)
            print(f"[apply] {'wrote' if changed else 'no change needed'}: {target}")
            return True

        def _do_check() -> bool:
            res = healthcheck.run_all()
            ok = bool(res[0]) if isinstance(res, tuple) else bool(res)
            print(f"[apply] post-check healthcheck: {'OK' if ok else 'FAIL'}")
            return ok

        return 0 if _apply_with_rollback("pre-apply auto-rollback (plan)", _do_apply, _do_check) else 1
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
    sp_plan.set_defaults(func=cmd_plan)
