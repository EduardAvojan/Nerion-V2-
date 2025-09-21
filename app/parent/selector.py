"""Provider selector for Nerion V2.

Legacy helpers queried local runtimes; in V2 we simply read provider IDs from
environment/config to keep downstream code paths working.
"""
from __future__ import annotations

from typing import Optional, Tuple
import os


def auto_select_model() -> Optional[Tuple[str, str, Optional[str]]]:
    provider_id = os.getenv("NERION_V2_CODE_PROVIDER") or os.getenv("NERION_V2_DEFAULT_PROVIDER")
    if not provider_id:
        return None
    if ":" in provider_id:
        provider, model = provider_id.split(":", 1)
    else:
        provider, model = provider_id, "default"
    return (provider, model, None)
