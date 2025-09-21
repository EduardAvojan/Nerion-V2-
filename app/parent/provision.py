"""No-op provisioning helpers for API-first Nerion V2."""
from __future__ import annotations

from typing import Tuple


def ensure_available(backend: str, model: str) -> Tuple[bool, str]:
    return True, f"provider-managed: {backend}:{model}"


def check_available(backend: str, model: str) -> bool:
    return True
