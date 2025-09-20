

from __future__ import annotations

import os
import json
import math
from typing import Optional, Dict, Any, Callable

from ops.security.net_gate import NetworkGate

# ---------------- Network preference store & status chip -------------------
_PREFS_PATH = os.path.join('out', 'policies', 'network_prefs.json')

_def_prefs: Dict[str, Any] = {
    "always_allow_by_task": {},   # e.g., {"web_search": true}
    "always_allow_by_domain": {}, # e.g., {"example.com": true}
    "offline_until": None,        # epoch seconds until which we remain offline
}


def load_net_prefs() -> Dict[str, Any]:
    """Load persisted network preferences; returns defaults if missing/invalid."""
    try:
        with open(_PREFS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data.setdefault('always_allow_by_task', {})
                return data
    except Exception:
        pass
    return json.loads(json.dumps(_def_prefs))


def save_net_prefs(prefs: Dict[str, Any]) -> None:
    """Persist network preferences (best-effort)."""
    try:
        os.makedirs(os.path.dirname(_PREFS_PATH), exist_ok=True)
        with open(_PREFS_PATH, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _fmt_time_left(secs: Optional[float]) -> str:
    if secs is None:
        return ''
    try:
        s = max(0, int(round(secs)))
        if s < 60:
            return f" ({s}s left)"
        if s < 3600:
            m = math.ceil(s / 60)
            return f" ({m}m left)"
        m = math.ceil(s / 60)
        h = m // 60
        rem = m % 60
        return f" ({h}h {rem}m left)" if rem else f" ({h}h left)"
    except Exception:
        return ''


def status_chip() -> str:
    """Human-readable online/offline chip with remaining time if online."""
    try:
        st = NetworkGate.state().name
        if st == 'SESSION':
            tl = NetworkGate.time_remaining()
            return '🌐 Online' + _fmt_time_left(tl)
        return '🔒 Offline'
    except Exception:
        return '🔒 Offline'


def ensure_network_for(
    task_type: str,
    speak: Callable[[str], None],
    listen_once: Callable[..., Optional[str]],
    *,
    url: Optional[str] = None,
    allow_always: bool = True,
    typed_fallback: bool = True,
    watcher=None,
) -> bool:
    """Unified helper to obtain a session grant for network use.

    - Respects master switch (selfcoder.config.allow_network).
    - Honors persisted "always allow for this task type" preferences.
    - If not already granted, prompts once for yes/no/always and grants a 10‑minute session.

    Callers should pass their `speak` and `listen_once` functions; `watcher` is
    accepted for signature symmetry but unused here.
    """
    try:
        from selfcoder.config import allow_network as _allow_net
    except Exception:
        def _allow_net() -> bool:  # type: ignore
            return True

    # Global master switch
    if not _allow_net():
        speak("Network is disabled by configuration.")
        return False

    # Load persisted preferences once
    try:
        prefs = load_net_prefs()
    except Exception:
        prefs = {"always_allow_by_task": {}, "always_allow_by_domain": {}, "offline_until": None}

    # Offline-only window in prefs suppresses prompts
    try:
        off_until = float((prefs or {}).get('offline_until') or 0.0)
    except Exception:
        off_until = 0.0
    import time as _t
    if off_until and _t.time() < off_until:
        speak("Offline mode is active for this session.")
        return False

    # Sticky preference: Always allow this task type
    if allow_always and bool((prefs.get('always_allow_by_task') or {}).get(task_type)):
        if not NetworkGate.can_use(task_type, url=url):
            try:
                NetworkGate.grant_session(task_types=None, minutes=10, domains=None, reason=f"auto {task_type}")
            except Exception:
                pass
        return True

    # Sticky preference: Always allow this domain
    dom = None
    if url:
        try:
            from urllib.parse import urlparse as _up
            dom = (_up(url).netloc or '').split(':')[0]
        except Exception:
            dom = None
    if allow_always and dom and bool((prefs.get('always_allow_by_domain') or {}).get(dom)):
        if not NetworkGate.can_use(task_type, url=url):
            try:
                NetworkGate.grant_session(task_types=None, minutes=10, domains=[dom], reason=f"auto domain {dom}")
            except Exception:
                pass
        return True

    # Session already granted?
    if NetworkGate.can_use(task_type, url=url):
        return True

    # Ask once
    prompt = (
        "I need internet access to do that. Allow web access for this session? "
        "Say 'yes', 'no', 'always' (task), or 'always domain' (remember this domain)."
    )
    speak(prompt)

    reply: Optional[str] = None
    try:
        reply = listen_once(timeout=10, phrase_time_limit=5)
    except Exception:
        reply = None

    if (not reply) and typed_fallback:
        try:
            txt = input("Type your response (yes/no/always): ")
            reply = (txt or '').strip()
        except Exception:
            reply = None

    if not reply:
        speak("Staying offline. Ask again if you change your mind.")
        return False

    r = str(reply).strip().lower()
    if r in {"y", "yes", "ok", "okay", "proceed", "allow"}:
        try:
            NetworkGate.grant_session(task_types=None, minutes=10, domains=None, reason=f"{task_type} session grant")
            note = f"Online for this session (10 minutes). {status_chip()}"
            speak(note)
            return True
        except Exception as e:
            speak(f"Couldn't enable network: {e}")
            return False
    if allow_always and r in {"always", "always allow", "remember", "always for this task", "always for this task type"}:
        try:
            prefs.setdefault('always_allow_by_task', {})[task_type] = True
            save_net_prefs(prefs)
            NetworkGate.grant_session(task_types=None, minutes=10, domains=None, reason=f"always {task_type}")
            note = f"I'll remember that. Online for this session (10 minutes). {status_chip()}"
            speak(note)
            return True
        except Exception as e:
            speak(f"Couldn't enable & remember: {e}")
            return False

    if allow_always and dom and r in {"always domain", "always for this domain", "remember domain"}:
        try:
            prefs.setdefault('always_allow_by_domain', {})[dom] = True
            save_net_prefs(prefs)
            NetworkGate.grant_session(task_types=None, minutes=10, domains=[dom], reason=f"always domain {dom}")
            note = f"I'll remember this domain. Online for this session (10 minutes). {status_chip()}"
            speak(note)
            return True
        except Exception as e:
            speak(f"Couldn't enable & remember domain: {e}")
            return False

    speak("Got it — remaining offline.")
    return False
