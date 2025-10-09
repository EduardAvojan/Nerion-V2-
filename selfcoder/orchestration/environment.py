"""
Environment and configuration utilities for orchestration.

This module provides utilities for environment variable management
and prompt preparation with smart defaults.
"""
import os
from pathlib import Path
from typing import Optional


def env_true(name: str, default: bool = False) -> bool:
    """Check if an environment variable is set to a truthy value.

    Args:
        name: Environment variable name
        default: Default value if not set

    Returns:
        True if the variable is set to "1", "true", "yes", or "on" (case-insensitive)
    """
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def prepare_for_prompt(instruction: str, file: Optional[Path] = None) -> None:
    """Project-manager style prep for a user prompt.

    - If NERION_MODE=user (default) set safe, helpful defaults:
      - Enable auto model selection (task-aware) and strict JSON planning by default
      - Try to auto-select a backend/model; if missing, print a concise consent message
    - If NERION_MODE=dev, do nothing (developer controls env).

    Args:
        instruction: The user instruction/prompt
        file: Optional file path context
    """
    mode = (os.getenv("NERION_MODE") or "user").strip().lower()
    if mode != "user":
        return
    # Defaults for user mode
    os.environ.setdefault("NERION_JSON_GRAMMAR", "1")
    os.environ.setdefault("NERION_LLM_STRICT", "1")
    default_provider = os.getenv("NERION_V2_CODE_PROVIDER") or os.getenv("NERION_V2_DEFAULT_PROVIDER") or "openai:gpt-5"
    os.environ.setdefault("NERION_V2_CODE_PROVIDER", default_provider)
