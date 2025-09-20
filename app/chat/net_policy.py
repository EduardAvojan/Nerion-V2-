"""Unified network policy façade for app code.

Bridges the global master switch (allow_network) and session-scoped grants
via NetworkGate into simple helpers used by routes/engine.
"""

from __future__ import annotations
from typing import Optional, Iterable
from selfcoder.config import allow_network
from ops.security.net_gate import NetworkGate


def can_use(task_type: str, *, url: Optional[str] = None) -> bool:
    """Return True if master switch allows and an active session covers the task/url."""
    if not allow_network():
        return False
    return NetworkGate.can_use(task_type, url=url)


def request_session(
    task_type: str,
    *,
    minutes: int = 10,
    url: Optional[str] = None,
    domains: Optional[Iterable[str]] = None,
    reason: str = "session grant",
) -> None:
    """Grant a session (raises PermissionError if master switch denies)."""
    # Note: url is unused here (domain scoping uses explicit domains)
    NetworkGate.grant_session(
        task_types=None if not task_type else [task_type],
        minutes=minutes,
        domains=list(domains or []),
        reason=reason,
    )

