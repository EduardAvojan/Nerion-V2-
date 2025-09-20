

import logging
from pathlib import Path

# Journaling helper (non-fatal on errors)
try:
    from .. import journal as _journal  # type: ignore
except Exception:  # pragma: no cover - journal is optional
    _journal = None  # type: ignore

logger = logging.getLogger(__name__)


def _journal_append(entry: dict) -> None:
    """Best-effort append to app/journal.jsonl (non-fatal on errors)."""
    try:  # pragma: no cover - we don't want tests to fail on journal IO
        if _journal and hasattr(_journal, "append"):
            _journal.append(entry)
    except Exception:
        pass


def run_self_coding_pipeline(instruction: str, speak, listen_once) -> bool:
    """
    Plan and apply code changes from a natural-language *instruction*.

    This module is UI-agnostic. The caller must provide:
      - `speak(text: str) -> None`    : to deliver short voice prompts
      - `listen_once(...) -> str|None`: to capture a single follow-up utterance

    Behavior:
      • Builds a plan from the NL instruction using the selfcoder planner.
      • If the plan does not include a target file, asks the user for one by voice
        and performs some light normalization of the spoken path.
      • If target is outside the repository, applies directly (no snapshot/HC).
      • If inside the repo, takes a snapshot → applies → runs healthcheck →
        rolls back on failure.

    Returns True on success, False otherwise.
    """
    _journal_append({"kind": "self_coding_start", "instruction": instruction})
    try:
        # Lazy imports to keep normal startup quick
        import os
        # Prefer LLM planner (Coder V2) when available; fall back to heuristics
        use_llm = (os.getenv('NERION_USE_CODER_LLM') or '1').strip().lower() in {'1','true','yes','on'}
        try:
            if use_llm:
                from selfcoder.planner.llm_planner import plan_with_llm as _plan_build
            else:
                from selfcoder.planner.planner import plan_edits_from_nl as _plan_build
        except Exception:
            from selfcoder.planner.planner import plan_edits_from_nl as _plan_build
        from selfcoder.orchestrator import apply_plan
        from selfcoder.healthcheck import run_all
        from selfcoder.vcs import git_ops

        # Build plan
        plan = _plan_build(instruction, None)
        if not plan or not plan.get("actions"):
            _journal_append({
                "kind": "self_coding_no_plan",
                "instruction": instruction,
                "reason": "no actions returned",
            })
            return False

        # Ask for a target if missing
        if not plan.get("target_file"):
            try:
                speak("Which file should I modify? For example: app slash nerion underscore chat dot pie.")
            except Exception:
                pass
            fp = listen_once(timeout=14, phrase_time_limit=12)
            if not fp:
                try:
                    speak("I couldn't catch a file path. Skipping.")
                except Exception:
                    pass
                _journal_append({
                    "kind": "self_coding_aborted",
                    "instruction": instruction,
                    "reason": "no target file via voice",
                })
                return False
            cleaned = (fp or "").strip()
            cleaned = cleaned.replace(" slash ", "/")
            cleaned = cleaned.replace(" dot ", ".")
            cleaned = cleaned.replace(" underscore ", "_")
            cleaned = cleaned.strip().replace("  ", " ")
            plan["target_file"] = cleaned

        target = Path(str(plan.get("target_file"))).expanduser().resolve()
        repo_root = Path(__file__).resolve().parents[2]  # project root (parent of app/)
        try:
            target.relative_to(repo_root)
            within_repo = True
        except Exception:
            within_repo = False

        # Outside repo: write directly without snapshot/healthcheck
        if not within_repo:
            try:
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("", encoding="utf-8")
            except Exception:
                return False
            apply_plan(plan, dry_run=False)
            logger.info("[SELF-CODING] applied outside-repo file (no healthcheck/rollback)")
            _journal_append({
                "kind": "self_coding_applied",
                "instruction": instruction,
                "target": str(target),
                "within_repo": False,
                "healthcheck": None,
                "rolled_back": False,
                "success": True,
            })
            return True

        # Inside repo: snapshot → apply → healthcheck → rollback on failure
        try:
            git_ops.snapshot(f"Voice self-coding: {instruction}")
        except Exception:
            pass

        changed = apply_plan(plan, dry_run=False)

        try:
            ok, _ = run_all()
        except Exception:
            ok = False

        if not ok:
            try:
                git_ops.restore_snapshot()
            except Exception:
                pass
            _journal_append({
                "kind": "self_coding_failed",
                "instruction": instruction,
                "target": str(target),
                "within_repo": True,
                "healthcheck": False,
                "rolled_back": True,
                "success": False,
            })
            return False

        _journal_append({
            "kind": "self_coding_applied",
            "instruction": instruction,
            "target": str(target),
            "within_repo": True,
            "healthcheck": True,
            "rolled_back": False,
            "success": True,
        })
        return bool(changed)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Self-coding pipeline failed")
        return False
