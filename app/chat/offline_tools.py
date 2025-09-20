from __future__ import annotations

from typing import Optional, List
import datetime
import re
import os

# Optional deps guarded so we never crash
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

# Reuse centralized time util for consistency
try:
    from ops.runtime.env import now_human
except Exception:  # pragma: no cover
    def now_human(_tz: Optional[str] = None) -> str:
        dt = datetime.datetime.now()
        return f"{dt:%Y-%m-%d %H:%M:%S}"

# ----------------------------------------------------------------------------
# Offline, low-latency tools (NO network)
# These functions are intentionally narrow and side-effect free. Each accepts
# an optional input string to be compatible with generic dispatchers.
# ----------------------------------------------------------------------------

__all__ = [
    "get_current_time",
    "get_current_date",
    "recall_memory",
    "pin_memory",
    "forget_memory",
    "set_timer",
    "stop_timer",
    "run_healthcheck",
    "run_diagnostics",
    "check_cpu_usage",
    "check_memory_usage",
    "perform_calculation",
    "set_volume",
    "get_agent_status",
]


# --- Time & Date Tools -------------------------------------------------------

def get_current_time(_: Optional[str] = None) -> str:
    """Return the current local time in a human-friendly format (no network)."""
    # now_human already formats with TZ where available
    human = now_human()
    # Provide a compact variant for chat UX
    try:
        # If now_human includes TZ, keep it; otherwise add a readable time
        dt = datetime.datetime.now()
        pretty = dt.strftime("%I:%M %p").lstrip("0")
        pretty = re.sub(r"\s+", " ", pretty).strip()
        return f"The current time is {pretty}."
    except Exception:
        return f"Current time: {human}"


def get_current_date(_: Optional[str] = None) -> str:
    """Return today's date in a human-friendly format (no network)."""
    dt = datetime.datetime.now()
    return f"Today's date is {dt:%A, %B %d, %Y}."


# --- Memory Management (offline) --------------------------------------------
try:
    from .memory_bridge import LongTermMemory  # type: ignore
except Exception:  # pragma: no cover
    LongTermMemory = None  # type: ignore


def recall_memory(query: Optional[str] = None) -> str:
    """Summarize or search long‑term memory without crashing if storage is absent.

    - No query: return a short summary of top memories.
    - With query: return up to 5 relevant facts.
    """
    if LongTermMemory is None:
        return "Memory store is unavailable."
    try:
        mem = LongTermMemory("memory_db.json")
        q = (query or "").strip()
        def _humanize(f: str) -> str:
            s = (f or "").strip()
            if not s:
                return s
            s = s.rstrip(" .")
            # Common transforms to address pronouns and templates
            s = re.sub(r"^user\s+likes\s+", "You like ", s, flags=re.I)
            s = re.sub(r"^user\s+preference:\s*", "Your preference: ", s, flags=re.I)
            s = re.sub(r"^user\s+role/work:\s*", "Your role/work: ", s, flags=re.I)
            s = re.sub(r"^user\s+prefers\s+to\s+be\s+called\s+", "You prefer to be called ", s, flags=re.I)
            s = re.sub(r"^user\s+lives\s+in\s+", "You live in ", s, flags=re.I)
            # Preserve the action verb (like/love/enjoy) and the rest of the phrase
            s = re.sub(
                r"^(?:i)\s+(?:really\s+)?(like|love|enjoy)\s+",
                lambda m: f"You {m.group(1)} ",
                s,
                flags=re.I,
            )
            # Capitalize first letter and add period
            s = s[0].upper() + s[1:] if s else s
            if not s.endswith('.'):
                s += '.'
            return s

        if not q:
            # Parse summarize_top output to facts list
            summary = mem.summarize_top(8) or ""
            if not summary:
                return "No long-term memories stored yet."
            lines: List[str] = [re.sub(r"^[-\s]+", "", ln).strip() for ln in summary.splitlines() if ln.strip()]
            human = [_humanize(ln) for ln in lines]
        else:
            facts = mem.find_relevant(q, k=5)
            if not facts:
                return "I couldn't find anything in memory for that."
            human = []
            for f in facts:
                fact_text = f.get('fact') if isinstance(f, dict) else f
                human.append(_humanize(fact_text))
        return "Here’s what I remember:\n" + "\n".join(f"- {h}" for h in human)
    except Exception as e:
        return f"Memory lookup failed: {e}"


def pin_memory(text: Optional[str] = None) -> str:
    """Pin a fact to long‑term memory. If text is omitted, ask the user to specify it."""
    if LongTermMemory is None:
        return "Memory store is unavailable."
    t = (text or "").strip()
    if not t:
        return "Please tell me what to pin (e.g., 'pin that I like jazz')."
    try:
        mem = LongTermMemory("memory_db.json")
        mem.add_fact(t if t.endswith(".") else t + ".", scope="long")
        return "I’ve saved that in my memory."
    except Exception as e:
        return f"Couldn't pin that: {e}"


def forget_memory(query: Optional[str] = None) -> str:
    """Forget a fact matching the query (best‑effort)."""
    if LongTermMemory is None:
        return "Memory store is unavailable."
    q = (query or "").strip()
    if not q:
        return "Please tell me what to forget."
    try:
        mem = LongTermMemory("memory_db.json")
        n, matched = mem.forget_smart(q, last_hint=None)
        if n:
            return f"Removed 1 memory item: {matched}."
        return "I couldn't find a matching memory to forget."
    except Exception as e:
        return f"Couldn't forget that: {e}"


# --- Timers (placeholders; deliberately no OS side-effects yet) --------------

def set_timer(duration_string: Optional[str] = None) -> str:
    """Stub for a timer; parsing scheduled for a later step."""
    return (
        "Timer feature isn't implemented yet. If you'd like, we can add a safe "
        "threaded timer with natural-language parsing next."
    )


def stop_timer(_: Optional[str] = None) -> str:
    """Stub for stopping a timer; scheduled for a later step."""
    return "Timer feature isn't implemented yet."


# --- System Status & Diagnostics --------------------------------------------

def run_healthcheck(_: Optional[str] = None) -> str:
    """Run internal healthcheck if available; never crashes if absent.

    selfcoder.healthcheck.run_all() returns a boolean; we format a concise result.
    """
    try:
        from selfcoder import healthcheck  # type: ignore
    except Exception:
        return "Healthcheck module is not available."
    try:
        ok = bool(healthcheck.run_all(verbose=False))
        status = "OK" if ok else "FAIL"
        return f"Healthcheck: {status}."
    except Exception as e:
        return f"Healthcheck failed to run: {e}"


def run_diagnostics(_: Optional[str] = None) -> str:
    """Lightweight diagnostics summary (CPU/MEM presence + recent error)."""
    parts = []
    # CPU
    if psutil is not None:
        try:
            cpu = psutil.cpu_percent(interval=0.2)
            parts.append(f"CPU {cpu:.0f}%")
        except Exception:
            parts.append("CPU n/a")
        try:
            mem = psutil.virtual_memory()
            parts.append(f"Mem {mem.percent:.0f}%")
        except Exception:
            parts.append("Mem n/a")
    else:
        parts.append("psutil not installed")
    # Include last error from action log if available
    try:
        log_path = os.path.join('out', 'experience', 'log.jsonl')
        if os.path.exists(log_path):
            import json
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-200:]
            last_err = None
            for ln in reversed(lines):
                try:
                    rec = json.loads(ln)
                    if not rec.get('outcome_success'):
                        last_err = (rec.get('action_taken') or {}).get('routed') or (rec.get('error') or 'error')
                        break
                except Exception:
                    continue
            if last_err:
                parts.append(f"last err: {last_err}")
    except Exception:
        pass
    return "Diagnostics: " + ", ".join(parts)


def check_cpu_usage(_: Optional[str] = None) -> str:
    """Return current system-wide CPU usage percentage if psutil present."""
    if psutil is None:
        return "CPU usage unavailable (psutil not installed)."
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        return f"Current CPU load is {cpu_percent:.0f}%."
    except Exception as e:
        return f"CPU usage unavailable: {e}"


def check_memory_usage(_: Optional[str] = None) -> str:
    """Return current system memory usage if psutil present."""
    if psutil is None:
        return "Memory usage unavailable (psutil not installed)."
    try:
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_used_gb = round(mem.used / (1024**3), 1)
        return f"Memory usage is {mem_percent:.0f}% ({mem_used_gb} GB used)."
    except Exception as e:
        return f"Memory usage unavailable: {e}"


# --- Utilities ---------------------------------------------------------------

def perform_calculation(expression: Optional[str] = None) -> str:
    """Placeholder for a safe evaluator; returns a friendly message for now."""
    return "Calculator is not implemented yet."


def set_volume(level: Optional[str] = None) -> str:
    """Placeholder. Cross-platform volume control will require platform hooks."""
    return "Volume control is not implemented yet."


def get_agent_status(_: Optional[str] = None) -> str:
    """Short status string; designed to be cheap and offline-only."""
    try:
        from ops.security.net_gate import NetworkGate  # type: ignore
        st = NetworkGate.state().name
        tl = None
        try:
            tl = NetworkGate.time_remaining()
        except Exception:
            tl = None
        chip = "🔒 Offline" if st != "SESSION" else ("🌐 Online" + (f" (~{int(tl//60)+1}m left)" if tl else ""))
        return f"Nerion is operational. Network: {chip}."
    except Exception:
        return "Nerion is operational."
