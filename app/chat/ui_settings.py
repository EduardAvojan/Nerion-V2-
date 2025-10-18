"""UI settings reader for real-time provider switching.

Reads from ~/.nerion/ui-settings.json to allow mid-session provider changes
without restarting the Python backend.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


def get_ui_settings_path() -> Path:
    """Return path to UI settings file."""
    return Path.home() / '.nerion' / 'ui-settings.json'


def read_ui_settings() -> Optional[Dict[str, Any]]:
    """Read UI settings from ~/.nerion/ui-settings.json.

    Returns None if file doesn't exist or is invalid.
    Called on every request for real-time updates.
    """
    settings_path = get_ui_settings_path()

    if not settings_path.exists():
        return None

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        return settings
    except (json.JSONDecodeError, IOError, OSError):
        return None


def get_provider_for_role(role: str) -> Optional[str]:
    """Get provider:model string for a specific role from UI settings.

    Args:
        role: One of 'chat', 'code', 'planner'

    Returns:
        Provider string like 'anthropic:claude-sonnet-4-5-20250929' or None
    """
    settings = read_ui_settings()

    if not settings or 'providers' not in settings:
        return None

    providers = settings['providers']

    if role not in providers:
        return None

    role_config = providers[role]

    if not isinstance(role_config, dict):
        return None

    provider = role_config.get('provider')
    model = role_config.get('model')

    if not provider or not model:
        return None

    return f"{provider}:{model}"


__all__ = ['get_ui_settings_path', 'read_ui_settings', 'get_provider_for_role']
