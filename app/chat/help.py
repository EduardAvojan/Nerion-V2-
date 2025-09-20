"""Offline help router for common Nerion topics.

Usage: get_help(None) or get_help("memory"). Returns a concise string.
"""

from __future__ import annotations

from typing import Optional


_TOPICS = {
    "general": (
        "Help – General\n"
        "- Say 'what can you do?' for a dynamic overview.\n"
        "- See docs/CHEATSHEET.md for a full command list.\n"
        "- Try: 'help voice', 'help memory', 'help files', 'help dev', 'help diagnostics'."
    ),
    "voice": (
        "Help – Voice\n"
        "- Hold SPACE to talk; release to stop (true PTT).\n"
        "- CapsLock (or F9/F12) toggles Speech ON/OFF.\n"
        "- '/speech on|off', '/voice on|off', '/mute on|off', '/say <text>'.\n"
        "- 'concise mode on|off' keeps general answers to one sentence."
    ),
    "memory": (
        "Help – Memory\n"
        "- 'remember that I like tea' stores a preference.\n"
        "- 'what do you remember about me' shows humanized recall.\n"
        "- 'forget <fact>' removes; 'unpin <fact>' demotes; 'pin that/this' pins last."
    ),
    "diagnostics": (
        "Help – Diagnostics\n"
        "- 'run health check' or 'diagnose system health' → healthcheck.\n"
        "- 'run diagnostics' → CPU/MEM summary.\n"
        "- 'run smoke tests' → quick pytest smoke."
    ),
    "files": (
        "Help – Files\n"
        "- 'read the file README.md' reads a repo‑jailed text file.\n"
        "- 'summarize file app/nerion_chat.py' creates a concise summary."
    ),
    "dev": (
        "Help – Developer Mode\n"
        "- 'switch to dev mode' to see capability options.\n"
        "- 'details for self-code' / 'details for security', etc.\n"
        "- 'exit dev mode' to return to normal."
    ),
    "planner": (
        "Help – Planner & Tools\n"
        "- Command verbs (run/scan/check/diagnose/update/test/benchmark/audit/format/lint/apply) plan tools.\n"
        "- Zero‑shot tools: list_plugins, run_pytest_smoke, run_diagnostics, read_file, summarize_file."
    ),
    "routing": (
        "Help – Routing & Safety\n"
        "- Offline‑first; web actions ask permission (allow/no/always).\n"
        "- Direct‑run: healthcheck/diagnostics/smoke with ack + spinner."
    ),
    "observability": (
        "Help – Observability\n"
        "- 'show last actions' shows recent tool calls with status/duration.\n"
        "- 'show last errors' shows the last few failures."
    ),
    "control": (
        "Help – Control\n"
        "- 'cancel' aborts a long multi‑step run between steps.\n"
        "- CapsLock toggles Speech; '/q' skips turn."
    ),
}


def help_topics() -> str:
    keys = ", ".join(sorted(_TOPICS))
    return (
        "Help – Topics\n"
        f"Available: {keys}.\n"
        "Try 'help <topic>' (e.g., 'help voice').\n"
        "See docs/CHEATSHEET.md for a full list."
    )


def get_help(topic: Optional[str]) -> str:
    if not topic:
        return help_topics()
    key = str(topic).strip().lower().replace(" ", "")
    # accept common aliases
    aliases = {
        "general": "general",
        "voice": "voice",
        "audio": "voice",
        "memory": "memory",
        "diagnostics": "diagnostics",
        "health": "diagnostics",
        "files": "files",
        "file": "files",
        "dev": "dev",
        "developer": "dev",
        "planner": "planner",
        "tools": "planner",
        "routing": "routing",
        "observability": "observability",
        "logs": "observability",
        "control": "control",
        "cancel": "control",
        "concise": "voice",
    }
    key = aliases.get(key, key)
    return _TOPICS.get(key, help_topics())

