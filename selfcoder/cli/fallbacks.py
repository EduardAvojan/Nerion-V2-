"""Fallback command implementations for when cli_ext modules are unavailable."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .simulation import maybe_simulate

try:
    from selfcoder import healthcheck
except Exception:
    healthcheck = None  # minimal fallback for shadow tests


def register_plan_fallback(sub):
    """Register fallback 'plan' command when cli_ext.plan is unavailable."""
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
                from selfcoder.planner.prioritizer import build_planner_context
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
            from selfcoder.planner.utils import (
                sanitize_plan,
                repo_fingerprint,
                load_plan_cache,
                save_plan_cache,
                attach_brief_metadata,
            )
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
                from selfcoder.planner.apply_policy import evaluate_apply_policy, apply_allowed
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


def register_health_fallback(sub):
    """Register fallback 'healthcheck' command when cli_ext.health is unavailable."""
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


def register_rename_fallback(sub):
    """Minimal fallback for 'rename' in shadow projects.
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
            return maybe_simulate(args, "rename", lambda: [])
        return 0

    sp.set_defaults(func=_run)
