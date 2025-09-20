

"""Session continuity helpers for Nerion (short‑term memory across restarts).

This module is self‑contained and does not import the runner to avoid cycles.
The runner should call `set_state_accessors(STATE, _auto_title)` once on init.
"""
from __future__ import annotations
import os
import time
import json
import atexit
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "_load_session_state_if_fresh",
    "_save_session_state",
    "set_state_accessors",
]

# Repo‑local storage
SESSION_STATE_PATH = Path('out') / 'session_state.json'
SESSION_TTL_S = int(os.getenv('NERION_SESSION_TTL_S', str(2 * 24 * 60 * 60)))  # default 48h

# Late‑bound accessors set by the runner to avoid import cycles
_STATE = None  # ChatState instance
_AUTO_TITLE: Optional[Callable[[str], str]] = None

def set_state_accessors(state_obj: Any, auto_title_fn: Callable[[str], str]) -> None:
    """Inject STATE and _auto_title from the runner (avoids circular import)."""
    global _STATE, _AUTO_TITLE
    _STATE = state_obj
    _AUTO_TITLE = auto_title_fn


def _save_session_state() -> None:
    """Persist the tail of the active conversation so we can restore it on next start."""
    try:
        if _STATE is None:
            return
        active = getattr(_STATE, 'active', None)
        if not active:
            return
        hist = list(getattr(active, 'chat_history', []) or [])
        topic = getattr(active, 'topic', '') or ''
        last_art = getattr(active, 'last_artifact_path', None)
        data = {
            'ts': time.time(),
            'topic': topic,
            'chat_tail': hist[-8:],
            'last_artifact_path': last_art,
        }
        SESSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSION_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        # never crash on exit
        pass


atexit.register(_save_session_state)


def _load_session_state_if_fresh() -> Tuple[bool, int]:
    """Load recent session state into STATE.active if the file is fresh.
    Returns (restored: bool, n_turns: int).
    """
    try:
        if _STATE is None or _AUTO_TITLE is None:
            return False, 0
        if not SESSION_STATE_PATH.exists():
            return False, 0
        with open(SESSION_STATE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ts = float(data.get('ts') or 0)
        if (time.time() - ts) > SESSION_TTL_S:
            return False, 0
        topic = data.get('topic') or ''
        tail = list(data.get('chat_tail') or [])
        last_art = data.get('last_artifact_path') or None
        # Open a new active conversation and append saved turns
        try:
            _STATE.open_new_conversation(topic=_AUTO_TITLE(topic or ''))
        except Exception:
            pass
        try:
            for t in tail:
                role = (t or {}).get('role')
                content = (t or {}).get('content')
                if role and content:
                    _STATE.append_turn(role, content)
        except Exception:
            pass
        try:
            if last_art:
                _STATE.set_last_artifact_path(last_art)
        except Exception:
            pass
        return True, len(tail)
    except Exception:
        return False, 0