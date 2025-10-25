"""Provider-based LLM router for Nerion V2.

In the API-first architecture we no longer probe local runtimes. Instead this router reads
provider identifiers from environment variables (or falls back to config defaults) and
returns normalized `(provider, model, base_url)` tuples for legacy callers.
"""
from __future__ import annotations

from typing import Optional, Tuple
import os

# Read defaults from model catalog, not hardcoded values
# Environment variables take precedence for temporary overrides
DEFAULT_CHAT_PROVIDER = os.getenv("NERION_V2_CHAT_PROVIDER") or os.getenv("NERION_V2_DEFAULT_PROVIDER")
DEFAULT_CODE_PROVIDER = os.getenv("NERION_V2_CODE_PROVIDER") or os.getenv("NERION_V2_DEFAULT_PROVIDER")


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
    """Return (provider, model, base_url) for the requested task.

    Priority:
    1. UI settings (~/.nerion/ui-settings.json) - allows mid-session switching
    2. Environment variables - CLI/script overrides
    3. model_catalog.yaml defaults - fallback

    Returns (None, None, None) to delegate to provider registry if no override found.
    """
    task_kind = (task or "code").lower()

    # Priority 1: Check UI settings first (for real-time switching)
    from app.chat.ui_settings import get_provider_for_role
    ui_provider_id = get_provider_for_role(task_kind)
    if ui_provider_id:
        provider, model = _split_provider(ui_provider_id)
        _log_decision(task_kind, {"provider": provider, "model": model, "source": "ui-settings"})
        return provider, model, None

    # Priority 2: Check environment variables
    if task_kind == "chat":
        provider_id = os.getenv("NERION_V2_CHAT_PROVIDER") or DEFAULT_CHAT_PROVIDER
        if not provider_id:
            # No env override - let provider registry use model_catalog.yaml defaults
            _log_decision("chat", {"provider": "catalog-default", "model": "catalog-default"})
            return None, None, None
        provider, model = _split_provider(provider_id)
        os.environ.setdefault("NERION_V2_CHAT_PROVIDER", provider_id)
        _log_decision("chat", {"provider": provider, "model": model, "source": "env"})
        return provider, model, None

    # Default to code role
    provider_id = os.getenv("NERION_V2_CODE_PROVIDER") or DEFAULT_CODE_PROVIDER
    if not provider_id:
        # No env override - let provider registry use model_catalog.yaml defaults
        _log_decision("code", {"provider": "catalog-default", "model": "catalog-default", "file": file})
        return None, None, None
    provider, model = _split_provider(provider_id)
    os.environ.setdefault("NERION_V2_CODE_PROVIDER", provider_id)
    _log_decision("code", {"provider": provider, "model": model, "file": file, "source": "env"})
    return provider, model, None


__all__ = ["apply_router_env"]
