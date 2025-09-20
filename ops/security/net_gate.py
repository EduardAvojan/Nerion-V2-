from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import os
import re
import threading
import time
from typing import Dict, Iterable, Optional, Set
from urllib.parse import urlparse


class NetState(Enum):
    OFFLINE = 0
    SESSION = 1  # allowed for the session (possibly time/idle limited)


@dataclass
class Grant:
    task_types: Set[str] = field(default_factory=set)  # empty => all types
    domains: Set[str] = field(default_factory=set)     # empty => all domains
    expires_at: Optional[float] = None                 # epoch seconds; None => no time limit
    last_used_at: Optional[float] = None               # for idle auto-revoke


class NetworkGate:
    """
    Centralized network permission gate.

    Design goals:
    - Offline by default, controlled and auditable opt-in.
    - Single prompt per session with optional stickiness by task type.
    - Optional domain scoping and idle-time auto revocation.
    - No external dependencies; safe to import anywhere.
    """
    @classmethod
    def time_remaining(cls) -> Optional[float]:
        """Return remaining seconds before network is auto-revoked, or None if no time limit.
        Considers both absolute expiry and idle-timeout; returns the smaller positive remainder.
        Returns 0 if already expired. OFFLINE returns 0.
        """
        with cls._lock:
            cls._auto_revoke_if_idle_or_expired()
            if cls._state != NetState.SESSION or not cls._grant:
                return 0.0
            now = time.time()
            candidates = []
            if cls._grant.expires_at is not None:
                candidates.append(max(0.0, cls._grant.expires_at - now))
            if cls._idle_revoke_after > 0 and cls._grant.last_used_at is not None:
                idle_left = (cls._idle_revoke_after - (now - cls._grant.last_used_at))
                candidates.append(max(0.0, idle_left))
            if not candidates:
                return None  # no time-based limits
            return float(min(candidates))
    

    _lock = threading.RLock()
    _inited: bool = False

    # Effective policy loaded from settings / environment
    _master_allow: bool = True  # settings.allow_network_access (default True)
    _idle_revoke_after: float = 15 * 60  # seconds
    _remember_by_task_type: bool = True
    _session_window_seconds: Optional[float] = None  # refresh window; set by grant_session(minutes)

    # Session state (process-local)
    _state: NetState = NetState.OFFLINE
    _grant: Optional[Grant] = None

    # Lightweight audit trail persisted under out/security_audit/
    _audit_path: Optional[str] = None

    @classmethod
    def init(cls, settings: Optional[Dict] = None) -> None:
        """Initialize policy from settings/env. Safe to call multiple times."""
        with cls._lock:
            s = settings or {}
            # Master switch precedence: explicit settings override environment; default True.
            if "allow_network_access" in s:
                cls._master_allow = bool(s.get("allow_network_access"))
            else:
                env_allow = os.getenv("NERION_ALLOW_NETWORK")
                if env_allow is not None:
                    cls._master_allow = env_allow.strip().lower() not in {"0", "false", "no"}
                else:
                    cls._master_allow = True

            cls._remember_by_task_type = bool(
                s.get("net", {}).get("remember_by_task_type", True)
            )

            idle_cfg = s.get("net", {}).get("idle_revoke_after", "15m")
            cls._idle_revoke_after = _parse_duration_seconds(idle_cfg, default_seconds=15 * 60)

            # Determine audit path lazily; do not create directories here to keep this side-effect free.
            cls._audit_path = s.get("paths", {}).get("net_audit_log")

            # If master switch is off, force offline state.
            if not cls._master_allow:
                cls._state = NetState.OFFLINE
                cls._grant = None

            cls._inited = True

    @classmethod
    def state(cls) -> NetState:
        with cls._lock:
            cls._auto_revoke_if_idle_or_expired()
            return cls._state

    @classmethod
    def can_use(cls, task_type: str, url: Optional[str] = None) -> bool:
        with cls._lock:
            cls._auto_revoke_if_idle_or_expired()
            if not cls._master_allow:
                return False
            if cls._state != NetState.SESSION or cls._grant is None:
                return False
            if cls._grant.task_types and task_type not in cls._grant.task_types:
                return False
            if url and cls._grant.domains:
                host = _host_of(url)
                if host not in cls._grant.domains:
                    return False
            return True

    @classmethod
    def grant_session(
        cls,
        *,
        task_types: Optional[Iterable[str]] = None,
        minutes: Optional[int] = None,
        domains: Optional[Iterable[str]] = None,
        reason: str = "",
    ) -> None:
        """Allow network for the current process/session.

        - task_types: restrict to these intents; None/empty => all intents.
        - minutes: time-based window; None => no absolute time limit (still subject to idle revoke).
        - domains: restrict to these hostnames; None/empty => all hosts.
        """
        with cls._lock:
            if not cls._master_allow:
                # Master switch (air-gapped) — cannot grant.
                raise PermissionError(
                    "Network is globally disabled by allow_network_access=false / NERION_ALLOW_NETWORK=0"
                )
            expires = time.time() + (minutes * 60) if minutes and minutes > 0 else None
            cls._session_window_seconds = (minutes * 60.0) if minutes and minutes > 0 else None
            cls._grant = Grant(
                task_types=set(task_types or []),
                domains=set(domains or []),
                expires_at=expires,
                last_used_at=time.time(),
            )
            cls._state = NetState.SESSION
            cls._audit("grant", {
                "reason": reason,
                "task_types": sorted(list(cls._grant.task_types)) or "*",
                "domains": sorted(list(cls._grant.domains)) or "*",
                "expires_at": cls._grant.expires_at,
            })

    @classmethod
    def revoke(cls, *, reason: str = "") -> None:
        with cls._lock:
            if cls._state != NetState.OFFLINE:
                cls._audit("revoke", {"reason": reason})
            cls._state = NetState.OFFLINE
            cls._grant = None

    @classmethod
    def assert_allowed(cls, *, task_type: str, url: Optional[str] = None) -> None:
        """Raise PermissionError when a network call is not permitted."""
        if cls.can_use(task_type, url=url):
            with cls._lock:
                # touch last_used_at and refresh sliding-window expiry
                if cls._grant:
                    now = time.time()
                    cls._grant.last_used_at = now
                    if cls._session_window_seconds:
                        cls._grant.expires_at = now + cls._session_window_seconds
            return
        # Build a helpful message explaining why we are blocked
        reasons = []
        with cls._lock:
            if not cls._master_allow:
                reasons.append("global policy deny (allow_network_access=false)")
            if cls._state != NetState.SESSION:
                reasons.append("no active session grant")
            elif cls._grant is None:
                reasons.append("missing grant")
            else:
                if cls._grant.task_types and task_type not in cls._grant.task_types:
                    reasons.append(f"task_type '{task_type}' not in allowed set")
                if url and cls._grant.domains:
                    host = _host_of(url)
                    if host not in cls._grant.domains:
                        reasons.append(f"domain '{host}' not in allowed set")
                if cls._grant.expires_at and time.time() >= cls._grant.expires_at:
                    reasons.append("grant expired")
                if (
                    cls._idle_revoke_after > 0
                    and cls._grant.last_used_at is not None
                    and (time.time() - cls._grant.last_used_at) >= cls._idle_revoke_after
                ):
                    reasons.append("grant idle-time revoked")
        raise PermissionError("Network access denied: " + "; ".join(reasons) or "unknown reason")

    # ---- helpers ---------------------------------------------------------

    @classmethod
    def _auto_revoke_if_idle_or_expired(cls) -> None:
        if cls._state != NetState.SESSION or cls._grant is None:
            return
        now = time.time()
        if cls._grant.expires_at and now >= cls._grant.expires_at:
            cls._audit("auto_revoke", {"reason": "expired"})
            cls._state = NetState.OFFLINE
            cls._grant = None
            return
        if (
            cls._idle_revoke_after > 0
            and cls._grant.last_used_at is not None
            and (now - cls._grant.last_used_at) >= cls._idle_revoke_after
        ):
            cls._audit("auto_revoke", {"reason": "idle_timeout"})
            cls._state = NetState.OFFLINE
            cls._grant = None

    @classmethod
    def _audit(cls, event: str, data: Dict) -> None:
        # Never raise from audit; best-effort only.
        try:
            payload = {
                "ts": time.time(),
                "event": event,
                "state": cls._state.name,
                **(data or {}),
            }
            # Write to file if configured and path exists; otherwise noop.
            path = cls._audit_path
            if not path:
                # Try a sensible default if the directory exists.
                default_dir = os.path.join("out", "security_audit")
                if os.path.isdir(default_dir):
                    path = os.path.join(default_dir, "net_gate.log")
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass


# -------------------------- utilities ------------------------------------

def _parse_duration_seconds(value, *, default_seconds: int) -> int:
    """Parse durations like "15m", "1h", "30s"; integers treated as seconds.
    Returns default_seconds on parse errors.
    """
    if value is None:
        return default_seconds
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower()
    m = re.fullmatch(r"(\d+)(s|m|h)?", s)
    if not m:
        return default_seconds
    num = int(m.group(1))
    unit = m.group(2) or "s"
    if unit == "s":
        return num
    if unit == "m":
        return num * 60
    if unit == "h":
        return num * 3600
    return default_seconds


def _host_of(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""
