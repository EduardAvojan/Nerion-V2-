"""Provider-based LLM router for Nerion V2.

In the API-first architecture we no longer probe local runtimes. Instead this router reads
provider identifiers from environment variables (or falls back to config defaults) and
returns normalized `(provider, model, base_url)` tuples for legacy callers.
"""
from __future__ import annotations

from typing import Optional, Tuple
import os

DEFAULT_CHAT_PROVIDER = os.getenv("NERION_V2_CHAT_PROVIDER") or os.getenv("NERION_V2_DEFAULT_PROVIDER") or "openai:o4-mini"
DEFAULT_CODE_PROVIDER = os.getenv("NERION_V2_CODE_PROVIDER") or os.getenv("NERION_V2_DEFAULT_PROVIDER") or "openai:o4-mini"


def _split_provider(provider_id: str) -> Tuple[str, str]:
    if ":" not in provider_id:
        return provider_id, "default"
    provider, model = provider_id.split(":", 1)
    return provider, model


def _log_decision(kind: str, payload: dict) -> None:
    if (os.getenv("NERION_ROUTER_VERBOSE") or "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[router] {kind}: {payload}")


def apply_router_env(
    *,
    instruction: Optional[str],
    file: Optional[str],
    task: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    """Return (provider, model, base_url) for the requested task."""
    task_kind = (task or "code").lower()
    if task_kind == "chat":
        provider_id = os.getenv("NERION_V2_CHAT_PROVIDER") or DEFAULT_CHAT_PROVIDER
        provider, model = _split_provider(provider_id)
        os.environ.setdefault("NERION_V2_CHAT_PROVIDER", provider_id)
        _log_decision("chat", {"provider": provider, "model": model})
        return provider, model, None

    provider_id = os.getenv("NERION_V2_CODE_PROVIDER") or DEFAULT_CODE_PROVIDER
    provider, model = _split_provider(provider_id)
    os.environ.setdefault("NERION_V2_CODE_PROVIDER", provider_id)
    _log_decision("code", {"provider": provider, "model": model, "file": file})
    return provider, model, None


__all__ = ["apply_router_env"]
