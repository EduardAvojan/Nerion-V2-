"""Compatibility shim for legacy imports expecting DeepSeekCoderV2."""
from __future__ import annotations

from typing import Optional

from .coder import Coder


class DeepSeekCoderV2(Coder):
    """Backwards-compatible alias that reuses the provider-backed coder."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(model=model, backend=None, base_url=base_url, temperature=temperature)


__all__ = ["DeepSeekCoderV2"]
