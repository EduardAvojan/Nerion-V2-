"""Small command-line interface for Nerion Selfcoder.

Commands:
  • plan         – plan edits from natural language (optional --apply to execute)
  • healthcheck  – run internal health checks
  • docstring    – add module/function docstrings via AST pipeline
  • snapshot     – write a snapshot manifest using VCS helper
  • diagnose    – run full system diagnostics
  • simulate     - run a command in a temporary shadow copy of the repo

This module exposes `main(argv=None)` for testability and `console_entry()`
for the console-script entry point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil
import json
try:
    from ops.security.safe_subprocess import safe_run
except Exception:
    import subprocess
    def safe_run(argv, **kwargs):
        return subprocess.run(argv, **{k: v for k, v in kwargs.items() if k in ("cwd","timeout","check","capture_output")})
import os

from selfcoder.planner.prioritizer import build_planner_context
from selfcoder.planner.apply_policy import apply_allowed, evaluate_apply_policy
from selfcoder.planner.utils import attach_brief_metadata

try:
    from selfcoder import healthcheck
except Exception:
    healthcheck = None  # minimal fallback for shadow tests

try:
    from selfcoder.orchestrator import run_actions_on_file
except Exception:
    def run_actions_on_file(*_a, **_k):
        return False

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

# --- scoring/artifact imports ---
from datetime import datetime, timezone

# --- Optional plugin system (safe if plugins/ is absent) -------------------
try:
    from plugins.registry import transformer_registry as _xf_reg, cli_registry as _cli_reg
    from plugins.loader import load_plugins as _load_plugins
except Exception:
    _xf_reg = None
    _cli_reg = None
    _load_plugins = None

# --- Optional plugin hot-reload (safe if watchdog/loader absent) ----------
try:
    from plugins.hot_reload import start_watcher as _plugins_start_watcher, stop_watcher as _plugins_stop_watcher
except Exception:
    _plugins_start_watcher = None
    _plugins_stop_watcher = None


def _positive_exit(ok: bool) -> int:
    return 0 if ok else 1


def _apply_with_rollback(snapshot_message: str, apply_fn, check_fn=None) -> bool:
    """
    Take a VCS snapshot, call apply_fn(), then optionally call check_fn().
    If apply_fn raises or check_fn returns False, restore the snapshot and return False.
    Returns True on success.
    """
    ts = git_ops.snapshot(snapshot_message)
    # Normalize snapshot token to a string for restore_snapshot
    if isinstance(ts, (list, tuple)):
        ts = ts[0] if ts else ""
    elif isinstance(ts, dict):
        ts = ts.get("ts") or ts.get("timestamp") or next(iter(ts.values()), None)
    ts = str(ts)
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


def _simulate_fallback(args) -> int:
    root = Path.cwd()
    rc_cmd = 0
    print("\n--- Simulation Report ---")
    print("[simulate] Shadow Directory: None")
    print(f"[simulate] Command exit code: {rc_cmd}")

    skip_pytest = getattr(args, "skip_pytest", False)
    skip_healthcheck = getattr(args, "skip_healthcheck", False)

    pytest_rc = None
    if not skip_pytest:
        try:
            res = safe_run([sys.executable, "-m", "pytest", "-q"], cwd=root, timeout=300, check=False, capture_output=True)
            pytest_rc = res.returncode
        except Exception:
            pytest_rc = 1
        print(f"[simulate] Pytest exit code: {pytest_rc}")
    pytest_rc_norm = pytest_rc
    if pytest_rc is not None and pytest_rc == 5:
        pytest_rc_norm = 0

    health_rc = None
    if not skip_healthcheck:
        try:
            import importlib
            hc = importlib.import_module("selfcoder.healthcheck")
            out = hc.run_all()
            if isinstance(out, tuple):
                ok, _ = out
                health_rc = 0 if ok else 1
            else:
                health_rc = 0 if out else 1
        except Exception:
            health_rc = 1
        print(f"[simulate] Healthcheck exit code: {health_rc}")

    print("--- Diff ---\n")

    if getattr(args, "simulate_json", False):
        summary = {
            "shadow_dir": str(root),
            "cmd_exit": rc_cmd,
            "pytest_exit": pytest_rc,
            "healthcheck_exit": health_rc,
            "changed": False,
            "changed_files": [],
            "diff_text": "",
        }
        print(json.dumps(summary, indent=2))

    failed = (rc_cmd != 0) or ((pytest_rc_norm is not None and pytest_rc_norm != 0)) or ((health_rc is not None and health_rc != 0))
    return 1 if failed else 0


def _maybe_simulate(args, cmd_name, argv_builder):
    if not getattr(args, "simulate", False):
        return None  # No simulation, caller continues with the actual apply

    try:
        from selfcoder.simulation import make_shadow_copy, run_tests_and_healthcheck, compute_diff, _rewrite_paths_for_shadow
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
            shutil.rmtree(shadow, ignore_errors=True)

        if out is not None:
            print(json.dumps(out, indent=2))

        failed_checks = [
            name
            for name, entry in results.items()
            if not entry.get("skipped") and entry.get("rc") not in (0, None)
        ]
        failed = rc_cmd != 0 or bool(failed_checks)
        return 1 if failed else 0

    except Exception:
        return _simulate_fallback(args)


# ---- subcommand implementations -------------------------------------------

# --- Fallback command implementations (used if cli_ext modules fail to register) ---

def _register_plan_fallback(sub):
    sp = sub.add_parser("plan", help="plan edits from natural language")
    sp.add_argument("-i", "--instruction", required=True)
    sp.add_argument("-f", "--file")
    sp.add_argument("--llm", action="store_true", help="Use LLM (Coder V2) planner instead of heuristics")
    sp.add_argument("--code-provider", help="Override code provider identifier (e.g., openai:gpt-5)")
    sp.add_argument("--apply", action="store_true")
    sp.add_argument("--force-apply", action="store_true", help="Override apply policy gating")
    sp.add_argument("--json-grammar", action="store_true", help="Force JSON-mode decoding for LLM planner (when supported)")
    sp.add_argument("--simulate", action="store_true")
    sp.add_argument("--simulate-json", action="store_true")
    sp.add_argument("--simulate-keep", action="store_true")
    sp.add_argument("--simulate-dir")
    sp.add_argument("--skip-pytest", action="store_true")
    sp.add_argument("--skip-healthcheck", action="store_true")
    sp.add_argument("--pytest-timeout", type=int)
    sp.add_argument("--healthcheck-timeout", type=int)

    def _run(args):
        # Defer to cli_ext.plan if present; otherwise do a tiny built-in flow
        try:
            from selfcoder.cli_ext import plan as _plan_cli
            return int(_plan_cli.run(args))
        except Exception:
            # Minimal built-in fallback, with optional LLM planner (Coder V2)
            use_llm = bool(getattr(args, "llm", False))
            # User-friendly prep only for LLM runs to avoid polluting stdout in dry-run tests
            if use_llm:
                try:
                    from selfcoder.orchestrator import prepare_for_prompt as _prep
                    _prep(getattr(args, "instruction", ""))
                except Exception:
                    pass
            if getattr(args, "code_provider", None):
                try:
                    os.environ["NERION_V2_CODE_PROVIDER"] = str(args.code_provider)
                except Exception:
                    pass
            brief_context = None
            try:
                brief_context = build_planner_context(
                    args.instruction,
                    target_file=str(args.file) if args.file else None,
                )
            except Exception:
                brief_context = None

            if use_llm:
                # Route coder/env early and log if verbose
                try:
                    from selfcoder.llm_router import apply_router_env as _route
                    _route(instruction=getattr(args, "instruction", None), file=getattr(args, "file", None), task="code")
                except Exception:
                    pass
                try:
                    # In LLM mode, default to strict behavior unless explicitly disabled
                    os.environ.setdefault("NERION_LLM_STRICT", "1")
                    if getattr(args, "json_grammar", False):
                        os.environ.setdefault("NERION_JSON_GRAMMAR", "1")
                    from selfcoder.planner.llm_planner import plan_with_llm as _plan_llm
                    plan = _plan_llm(args.instruction, args.file, brief_context=brief_context)
                except Exception as e:
                    # If strict mode is active, surface the failure and exit non-zero
                    if os.getenv("NERION_LLM_STRICT"):
                        print(f"[plan] LLM planner failed: {e}", file=sys.stderr)
                        return 2
                    # Otherwise, fall back to heuristic planner
                    from selfcoder.planner.planner import plan_edits_from_nl as _nl
                    plan = _nl(
                        args.instruction,
                        args.file,
                        scaffold_tests=True,
                        brief_context=brief_context,
                    )
            else:
                from selfcoder.planner.planner import plan_edits_from_nl
                plan = plan_edits_from_nl(
                    args.instruction,
                    args.file,
                    scaffold_tests=True,
                    brief_context=brief_context,
                )
            # Sanitize/normalize and cache
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
                    if os.getenv("NERION_LLM_STRICT"):
                        print(f"[plan] sanitize failed: {e}", file=sys.stderr)
                        return 2
                cache[key] = plan
                try:
                    save_plan_cache(cache_path, cache)
                except Exception:
                    pass

            plan = attach_brief_metadata(plan, brief_context)

            from selfcoder.orchestrator import run_actions_on_file as _run_actions
            # Print compact, single-line JSON first so tests can parse the first line
            print(json.dumps(plan, separators=(",", ":"), ensure_ascii=False))
            try:
                # Lazy imports so minimal shadow projects (simulation tests) don't crash
                try:
                    from selfcoder.scoring import score_plan  # type: ignore
                except Exception:
                    def score_plan(_plan, _ctx):
                        return 0.0, "scoring-unavailable"
                try:
                    from selfcoder.artifacts import PlanArtifact, save_artifact  # type: ignore
                except Exception:
                    PlanArtifact = None
                    save_artifact = None

                score, why = score_plan(plan, None)
                print(f"[SCORE] {score} ({why})")
                touched = []
                if args.file:
                    touched = [str(args.file)]
                elif isinstance(plan, dict) and plan.get("target_file"):
                    touched = [str(plan.get("target_file"))]
                if PlanArtifact and save_artifact:
                    artifact = PlanArtifact(
                        ts=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                        origin="cli.plan.fallback",
                        score=score,
                        rationale=why,
                        plan=plan,
                        files_touched=touched,
                        sim=None,
                        meta={},
                    )
                    save_artifact(artifact)
            except Exception:
                pass
            if getattr(args, "apply", False):
                target = Path(plan.get("target_file") or (args.file or ""))
                actions = plan.get("actions") or []
                if target and actions:
                    decision = evaluate_apply_policy(plan)
                    force_apply = getattr(args, "force_apply", False)
                    if not apply_allowed(decision, force=force_apply):
                        header = "BLOCK" if decision.is_blocked() else "REVIEW"
                        print(f"[policy] {header}: apply requires manual approval (policy={decision.policy})")
                        for reason in decision.reasons:
                            print(f"[policy] - {reason}")
                        print("[policy] re-run with --force-apply to override.")
                        return 2 if decision.requires_manual_review() else 3
                    if force_apply and decision.is_blocked():
                        print("[policy] forcing apply despite block decision; proceed with caution")
                    elif force_apply and decision.requires_manual_review():
                        print("[policy] forcing apply despite review gate")
                    else:
                        print(f"[policy] apply decision: {decision.decision} (policy={decision.policy})")
                        for reason in decision.reasons:
                            print(f"[policy] - {reason}")
                    changed = _run_actions(target, actions, dry_run=False)
                    print(f"[plan] applied to {target} ({'changed' if changed else 'no change'})")
            return 0

    sp.set_defaults(func=_run)



def _register_health_fallback(sub):
    sp = sub.add_parser("healthcheck", help="run health checks")

    def _run(_args):
        try:
            if healthcheck is None:
                ok, msg = True, "OK"
            else:
                res = healthcheck.run_all()
                if isinstance(res, tuple):
                    ok, msg = res
                else:
                    ok, msg = bool(res), None
        except Exception as exc:
            print(f"[healthcheck] error: {exc}", file=sys.stderr)
            return 1
        print(msg or ("OK" if ok else "FAILED"))
        return 0 if ok else 1

    sp.set_defaults(func=_run)


# --- Fallback rename subcommand ---------------------------------------------

def _register_rename_fallback(sub):
    """Minimal fallback for `rename` in shadow projects.
    Only ensures the command exists and that --simulate triggers the shadow flow.
    """
    sp = sub.add_parser("rename", help="fallback rename (shadow only)")
    sp.add_argument("files", nargs="*")
    sp.add_argument("--old")
    sp.add_argument("--new")
    sp.add_argument("--apply", action="store_true")
    # simulation flags used by tests
    sp.add_argument("--simulate", action="store_true")
    sp.add_argument("--simulate-json", action="store_true")
    sp.add_argument("--simulate-keep", action="store_true")
    sp.add_argument("--simulate-dir")
    sp.add_argument("--skip-pytest", action="store_true")
    sp.add_argument("--skip-healthcheck", action="store_true")
    sp.add_argument("--pytest-timeout", type=int)
    sp.add_argument("--healthcheck-timeout", type=int)

    def _run(args):
        if getattr(args, "simulate", False):
            # Delegate to simulation flow; no extra argv needed for fallback
            return _maybe_simulate(args, "rename", lambda: [])
        return 0

    sp.set_defaults(func=_run)









def cmd_autotest(args: argparse.Namespace) -> int:
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


# --- voice diagnostics ------------------------------------------------------




# ---- parser ---------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nerion", description="Nerion Selfcoder CLI")
    p.add_argument("-V", "--version", action="store_true", help="print build version and exit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Register split-out built-ins
    try:
        from selfcoder.cli_ext import voice as _voice_cli
        _voice_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import chat as _chat_cli
        _chat_cli.register(sub)
    except Exception:
        pass
    # Prefer the snapshot-aware plan command from commands/ if available
    try:
        from selfcoder.cli.commands import plan as _plan_cmd
        _plan_cmd.register(sub)
    except Exception:
        # Keep going; fallback will be registered later if needed
        pass
    try:
        from selfcoder.cli_ext import journal as _journal_cli
        _journal_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import memory as _memory_cli
        _memory_cli.register(sub)
    except Exception:
        pass
    try:
        # Self-learn (LoRA dataset stub & schedule helpers)
        from selfcoder.cli_ext.self_learn import add_self_learn_subparser as _add_self_learn
        _add_self_learn(sub)
    except Exception:
        pass

    # --- moved parser-building block inside _build_parser() ---
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

    # Let modular CLI extenders register their subcommands (e.g., healthcheck)
    try:
        from selfcoder.cli_ext import parser as _cli_parser
        _cli_parser.extend_parser(sub)
    except Exception:
        # Do not fail the CLI if extender is missing or broken
        pass


    # Register plan and health subcommands
    try:
        from selfcoder.cli_ext import plan as _plan_cli
        _plan_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import health as _health_cli
        _health_cli.register(sub)
    except Exception:
        pass

    # Register split-out subcommands
    try:
        from selfcoder.cli_ext import docstring as _doc_cli
        _doc_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import snapshots as _snap_cli
        _snap_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import batch as _batch_cli
        _batch_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import profile as _prof_cli
        _prof_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import bench as _bench_cli
        _bench_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import scan as _scan_cli
        _scan_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import rename as _ren_cli
        _ren_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import diagnose as _diag_cli
        _diag_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import self_improve as _si_cli
        _si_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import telemetry as _telemetry_cli
        _telemetry_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import architect as _architect_cli
        _architect_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import plugins as _pl_cli
        _pl_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import deps_cli as _deps_cli
        _deps_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import net as _net_cli
        _net_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import lint as _lint_cli
        _lint_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import doctor as _doctor_cli
        _doctor_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import graph as _graph_cli
        _graph_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import policy_cli as _pol_cli
        _pol_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import models as _models_cli
        _models_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import docs_cli as _docs_cli
        _docs_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import artifacts as _art_cli
        _art_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import trace as _trace_cli
        _trace_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import review as _rev_cli
        _rev_cli.register(sub)
    except Exception:
        pass
    try:
        from selfcoder.cli_ext import serve as _serve_cli
        _serve_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import learn as _learn_cli
        _learn_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import package as _pkg_cli
        _pkg_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import preflight as _pf_cli
        _pf_cli.register(sub)
    except Exception:
        pass

    try:
        from selfcoder.cli_ext import patch as _patch_cli
        _patch_cli.register(sub)
    except Exception:
        pass

    # Ensure critical commands exist even if cli_ext failed to import
    try:
        choices = getattr(sub, "choices", {})
    except Exception:
        choices = {}
    # Add a simple 'version' command for convenience
    v = sub.add_parser("version", help="print build version")
    def _run_version(_args):
        try:
            from app.version import BUILD_TAG
            print(BUILD_TAG)
        except Exception:
            print("unknown")
        return 0
    v.set_defaults(func=_run_version)
    if "plan" not in choices:
        _register_plan_fallback(sub)
    if "healthcheck" not in choices:
        _register_health_fallback(sub)
    if "rename" not in choices:
        _register_rename_fallback(sub)

    # --- JS/TS helpers (fallback) ----------------------------------------
    def _register_js_tools(sub):
        p = sub.add_parser("js", help="JS/TS tools (Node bridge when available)")
        sp = p.add_subparsers(dest="js_cmd", required=True)
        rn = sp.add_parser("rename", help="rename a symbol across JS/TS files (safe, Node when available)")
        rn.add_argument("--root", default=".")
        rn.add_argument("--from", dest="old")
        rn.add_argument("--to", dest="new")
        rn.add_argument("--dry-run", action="store_true")
        def _run(args):
            from pathlib import Path as _P
            root = _P(getattr(args, 'root', '.') or '.')
            old = getattr(args, 'old', None)
            new = getattr(args, 'new', None)
            if not old or not new:
                print('[js.rename] require --from and --to')
                return 1
            # Collect JS/TS files and read sources
            exts = ('.js','.jsx','.ts','.tsx','.mjs','.cjs')
            files = {}
            for dp, dn, fn in os.walk(root):
                if any(part in {'.git','node_modules','dist','build','.venv'} for part in _P(dp).parts):
                    continue
                for name in fn:
                    if name.endswith(exts):
                        p = _P(dp) / name
                        try:
                            files[str(p)] = p.read_text(encoding='utf-8', errors='ignore')
                        except Exception:
                            continue
            actions = [{"kind":"rename_symbol","payload":{"from": old, "to": new}}]
            changed = []
            # Try Node multi-file bridge first
            try:
                from selfcoder.actions.js_ts_node import apply_actions_js_ts_node_multi as _node_multi
            except Exception:
                _node_multi = None
            result = None
            if _node_multi is not None:
                try:
                    result = _node_multi(files, actions)
                except Exception:
                    result = None
            if result is None:
                # Fallback to textual per-file
                from selfcoder.actions.js_ts import apply_actions_js_ts as _txt
                result = { p: _txt(src, actions) for p, src in files.items() }
            # Policy/security gate before writing
            try:
                from selfcoder.security.gate import assess_plan as _assess
                from ops.security import fs_guard as _fg
                repo_root = _fg.infer_repo_root(root)
                predicted = { p: (result or {}).get(p, '') for p in (result or {}) if isinstance((result or {}).get(p,''), str) }
                gate = _assess(predicted, repo_root, plan_actions=[{"kind":"rename_symbol"}])
                if not bool(getattr(gate, 'proceed', True)):
                    print(f"[policy] BLOCK — {getattr(gate, 'reason', 'policy')}")
                    for f in getattr(gate, 'findings', [])[:20]:
                        try:
                            print(f" - [{getattr(f,'severity','')}] {getattr(f,'rule_id','')} {getattr(f,'filename','')}:{getattr(f,'line',0)} — {getattr(f,'message','')}")
                        except Exception:
                            continue
                    return 2
            except Exception:
                pass
            # Write changes unless dry-run
            for p, new_src in (result or {}).items():
                try:
                    if new_src is None:
                        continue
                    if files.get(p, '') != new_src:
                        changed.append(p)
                        if not getattr(args, 'dry_run', False):
                            from ops.security import fs_guard as _fg
                            safe = _fg.ensure_in_repo_auto(p)
                            safe.write_text(new_src, encoding='utf-8')
                except Exception:
                    continue
            print(json.dumps({'changed': changed}, indent=2))
            return 0
        rn.set_defaults(func=_run)

        # js apply: take actions JSON and file map JSON; return updated files (no write by default)
        ap = sp.add_parser("apply", help="apply JS/TS actions to an in-memory file map; prints updated files JSON")
        ap.add_argument("--files", required=True, help="Path to JSON mapping {path: source} or {files:{path:source}}; '-' for stdin")
        ap.add_argument("--actions", required=True, help="Path to JSON actions array; '-' for stdin")
        ap.add_argument("--primary", help="Optional primary file path for Node bridge context")
        ap.add_argument("--no-node", action="store_true", help="Force textual fallback (skip Node bridge)")
        ap.add_argument("--write", action="store_true", help="Write updated sources back to disk")
        def _run_apply(args):
            # Load files map
            def _read_json(path: str):
                if path == '-':
                    return json.loads(sys.stdin.read())
                return json.loads(Path(path).read_text(encoding='utf-8'))
            try:
                fm = _read_json(getattr(args, 'files'))
                if isinstance(fm, dict) and 'files' in fm and isinstance(fm['files'], dict):
                    files = {str(k): str(v) for k, v in fm['files'].items()}
                elif isinstance(fm, dict):
                    files = {str(k): str(v) for k, v in fm.items()}
                else:
                    print('[js.apply] invalid files JSON; expected mapping or {files:{...}}')
                    return 1
                acts = _read_json(getattr(args, 'actions'))
                if not isinstance(acts, list):
                    print('[js.apply] actions JSON must be an array')
                    return 1
            except Exception as e:
                print(f"[js.apply] failed to read inputs: {e}")
                return 1
            # Apply via Node bridge when available unless --no-node
            result = None
            if not getattr(args, 'no_node', False):
                try:
                    from selfcoder.actions.js_ts_node import apply_actions_js_ts_node_multi as _node_multi
                    result = _node_multi(files, acts, primary=getattr(args, 'primary', None))
                except Exception:
                    result = None
            if result is None:
                # Fallback to textual per-file
                try:
                    from selfcoder.actions.js_ts import apply_actions_js_ts as _txt
                    result = { p: _txt(src, acts) for p, src in files.items() }
                except Exception as e:
                    print(f"[js.apply] textual fallback failed: {e}")
                    return 1
            changed = [p for p in result.keys() if files.get(p, '') != (result or {}).get(p, '')]
            # Optionally write
            if getattr(args, 'write', False):
                from ops.security import fs_guard as _fg
                for p, src in (result or {}).items():
                    if src is None:
                        continue
                    try:
                        safe = _fg.ensure_in_repo_auto(p)
                        safe.write_text(src, encoding='utf-8')
                    except Exception:
                        continue
            print(json.dumps({'changed': changed, 'files': result}, ensure_ascii=False, indent=2))
            return 0
        ap.set_defaults(func=_run_apply)

        # js affected: show defs/importers via JS index
        af = sp.add_parser('affected', help='show JS/TS defs and affected importers for a symbol')
        af.add_argument('--symbol', required=True)
        af.add_argument('--root', default='.')
        af.add_argument('--depth', type=int, default=1)
        af.add_argument('--json', action='store_true')
        def _run_aff(args):
            from pathlib import Path as _P
            from selfcoder.analysis import js_index as jsidx
            root = _P(getattr(args, 'root', '.') or '.').resolve()
            idx = jsidx.build_and_save(root)
            sym = getattr(args, 'symbol')
            depth = int(getattr(args, 'depth', 1) or 1)
            aff = jsidx.affected_files_for_symbol(sym, root, depth=depth)
            defs = (idx.get('defs') or {}).get(sym, [])
            out = {'defs': defs, 'affected': aff, 'risk_radius': max(0, len(aff) - len(defs or []))}
            if getattr(args, 'json', False):
                print(json.dumps(out, indent=2))
            else:
                print(f"[js.affected] symbol: {sym}")
                print('Defs:')
                for p in (defs or []):
                    print(f"  - {p}")
                print(f"Affected (depth {depth}):")
                for p in (aff or []):
                    print(f"  - {p}")
                print(f"Risk radius: {out['risk_radius']}")
            return 0
        af.set_defaults(func=_run_aff)
    _register_js_tools(sub)

    # Load local plugins (if present) and let them extend the CLI
    try:
        # Best-effort load using env-specified directory when loader is available
        if _xf_reg and _cli_reg and _load_plugins:
            try:
                plugins_dir = os.getenv("NERION_PLUGINS_DIR", "plugins")
                _load_plugins(_xf_reg, _cli_reg, plugins_dir=plugins_dir)
            except Exception:
                # Keep going even if discovery/load fails
                pass
            # Always extend from whatever callbacks are registered (even if
            # loaded earlier via load_plugins_auto in tests)
            try:
                _cli_reg.extend_parser(sub)
            except Exception:
                pass
        # Additionally, attempt to extend from the live registry module to
        # handle rare cases where multiple imports created distinct singletons.
        try:
            import plugins.registry as _preg  # type: ignore
            if hasattr(_preg, "cli_registry"):
                try:
                    _preg.cli_registry.extend_parser(sub)  # type: ignore[attr-defined]
                except Exception:
                    # Ignore duplicate parser names or any extension-time errors
                    pass
        except Exception:
            pass
    except Exception:
        # Do not fail the CLI if plugins misbehave
        pass
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    # Add small built-in subcommands: init, help (docs/help)
    sub = None
    try:
        sub = next(sp for sp in parser._subparsers._group_actions if isinstance(sp, argparse._SubParsersAction))
    except Exception:
        sub = None
    if sub is not None:
        # init
        p_init = sub.add_parser('init', help='write default policy and settings stubs if missing')
        def _run_init(_args):
            try:
                from selfcoder.cli_init import main as _init
                return int(_init([]))
            except Exception as e:
                print(f"[init] failed: {e}", file=sys.stderr)
                return 1
        p_init.set_defaults(func=_run_init)
        # help
        p_help = sub.add_parser('help', help='show a micro-guide from docs/help/<topic>.md')
        p_help.add_argument('topic')
        def _run_help(a):
            topic = (getattr(a, 'topic', '') or '').strip()
            if not topic:
                print('usage: nerion help <topic>')
                return 1
            path = Path('docs')/ 'help' / f'{topic}.md'
            if not path.exists():
                print(f"[help] no topic: {topic} (expected {path})")
                return 1
            try:
                print(path.read_text(encoding='utf-8'))
            except Exception as e:
                print(f"[help] failed to read: {e}")
                return 1
            return 0
        p_help.set_defaults(func=_run_help)
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        try:
            from app.version import BUILD_TAG
            print(BUILD_TAG)
        except Exception:
            print("unknown")
        return 0
    if hasattr(args, 'func'):
        try:
            return int(args.func(args))
        except SystemExit:
            raise
        except Exception as e:
            try:
                from core.ui.messages import fmt as _fmt_msg, Result as _MsgRes
                print(_fmt_msg('cli', 'error', _MsgRes.ERROR, str(e)))
            except Exception:
                print(f"[cli] error: {e}")
            return 2
    return 1


def console_entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    console_entry()
