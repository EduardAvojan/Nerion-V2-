from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from app.chat.voice_io import safe_speak, listen_once

# Self‑improvement plumbing
from selfcoder.selfaudit import generate_improvement_plan
from selfcoder.orchestrator import apply_plan
from selfcoder.analysis.knowledge import index as kb_index
from selfcoder.planner.apply_policy import apply_allowed, evaluate_apply_policy
from selfcoder.governor import evaluate as governor_evaluate
from selfcoder.governor import note_execution as governor_note_execution

STATE_PATH = Path("out/policies/upgrade_state.json")
_OFFER_COOLDOWN_S = 45  # avoid repeating the prompt too frequently


def _now() -> float:
    return time.time()


def _load_state() -> Dict[str, Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _count_recent_knowledge(since_ts: Optional[float]) -> int:
    try:
        entries = kb_index.load_index()
        if not since_ts:
            return len(entries)
        return sum(1 for e in (entries or []) if int(e.get("date", 0)) > int(since_ts))
    except Exception:
        return 0


def _tonight_timestamp(now: float | None = None) -> float:
    t = time.localtime(now or _now())
    # schedule at 02:00 local next day
    next_day = time.mktime((t.tm_year, t.tm_mon, t.tm_mday + 1, 2, 0, 0, 0, 0, -1))
    return float(next_day)


def readiness_report(threshold: int = 5) -> Dict[str, Any]:
    """Compute simple readiness metrics and whether we should offer an upgrade."""
    st = _load_state()
    last = float(st.get("last_upgrade_ts", 0) or 0)
    last_offer = float(st.get("last_offer_ts", 0) or 0)
    recent_kb = _count_recent_knowledge(last)
    snooze_until = float(st.get("snooze_until", 0) or 0)
    scheduled_ts = float(st.get("scheduled_ts", 0) or 0)
    now = _now()
    return {
        "recent_knowledge": recent_kb,
        "threshold": threshold,
        "should_offer": (
            (recent_kb >= threshold)
            and (now >= snooze_until)
            and (not scheduled_ts or now >= scheduled_ts)
            and ((now - last_offer) >= _OFFER_COOLDOWN_S)
        ),
        "snooze_until": snooze_until,
        "scheduled_ts": scheduled_ts,
        "last_upgrade_ts": last,
        "last_offer_ts": last_offer,
    }


def handle_choice(reply: str, watcher=None) -> bool:
    """Handle a spoken/text choice for the upgrade prompt.

    Returns True if a recognized option was handled; False otherwise.
    Safe to call from the main chat loop (PTT or text input).
    """
    if not reply:
        return False
    r = str(reply).strip().lower()
    st = _load_state()

    if r in {"upgrade now", "now", "upgrade", "do it", "proceed"}:
        try:
            plan = generate_improvement_plan(Path("."))
            decision = evaluate_apply_policy(plan)

            def _env_bool(name: str) -> bool:
                val = os.getenv(name)
                if val is None:
                    return False
                return val.strip().lower() in {"1", "true", "yes", "on"}

            force = _env_bool("NERION_UPGRADE_FORCE")
            allow_review = _env_bool("NERION_UPGRADE_ALLOW_REVIEW")
            if not apply_allowed(decision, allow_review=allow_review, force=force):
                label = "blocked" if decision.is_blocked() else "requires review"
                reasons = ", ".join(decision.reasons) or "no reasons provided"
                note = f"Upgrade {label}: {reasons}."
                print("Nerion:", note)
                try:
                    safe_speak(note, watcher)
                except Exception:
                    pass
                return True

            if force and decision.is_blocked():
                print("[policy] forcing upgrade despite block decision")
            elif force and decision.requires_manual_review():
                print("[policy] forcing upgrade despite review gate")

            try:
                governor_decision = governor_evaluate(
                    "upgrade_agent.apply",
                    override=force,
                )
            except Exception:
                governor_decision = None

            if governor_decision and governor_decision.is_blocked():
                note = "Upgrade postponed: governor limits active."
                details = ", ".join(governor_decision.reasons)
                if details:
                    note += f" ({details})"
                if governor_decision.next_allowed_local:
                    note += f" Next window {governor_decision.next_allowed_local}."
                print("Nerion:", note)
                try:
                    safe_speak(note, watcher)
                except Exception:
                    pass
                return True

            if governor_decision and governor_decision.override_used:
                print("[governor] override flag detected; proceeding under manual override")

            governor_note_execution("upgrade_agent.apply")
            apply_plan(plan, preview=False)
            st["last_upgrade_ts"] = _now()
            st.pop("scheduled_ts", None)
            note = "Upgrade complete."
        except Exception as e:
            note = f"Upgrade failed: {e}"
        print("Nerion:", note)
        try:
            safe_speak(note, watcher)
        except Exception:
            pass
        _save_state(st)
        return True

    if r in {"remind me later", "later", "remind later"}:
        st["snooze_until"] = _now() + 2 * 3600
        _save_state(st)
        note = "Okay, I’ll remind you later."
        print("Nerion:", note)
        try:
            safe_speak(note, watcher)
        except Exception:
            pass
        return True

    if r in {"tonight", "tonite"}:
        st["scheduled_ts"] = _tonight_timestamp()
        _save_state(st)
        note = "Scheduled for tonight."
        print("Nerion:", note)
        try:
            safe_speak(note, watcher)
        except Exception:
            pass
        return True

    return False


def maybe_offer_upgrade(watcher=None, *, threshold: int = 5, ptt_mode: bool | None = None) -> None:
    """If sufficient new knowledge accumulated and not snoozed, offer a self‑learning upgrade.

    Offers a voice prompt: Upgrade now / Remind me later / Tonight.
    - Now: generate improvement plan and apply immediately.
    - Remind me later: snooze for 2 hours.
    - Tonight: schedule for 02:00 local next day.
    """
    rep = readiness_report(threshold=threshold)
    if not rep.get("should_offer"):
        return

    msg = "Self Learning Upgrade Available. Say 'upgrade now', 'remind me later', or 'tonight'."
    print("Nerion:", msg)
    try:
        safe_speak(msg, watcher)
    except Exception:
        pass
    # Record offer time to enforce cooldown and avoid spamming
    try:
        st = _load_state()
        st["last_offer_ts"] = _now()
        _save_state(st)
    except Exception:
        pass

    # In PTT mode, do not block on capture here; the main loop will handle choices.
    if ptt_mode:
        return

    # Otherwise, capture a quick response inline
    reply = None
    try:
        reply = listen_once(timeout=10, phrase_time_limit=5)
    except Exception:
        reply = None
    if not reply:
        return
    # Delegate to the shared handler
    handle_choice(str(reply), watcher)
    return
