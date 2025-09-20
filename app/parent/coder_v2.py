"""
DeepSeek Coder V2 adapter for Nerion (local via Ollama or compatible endpoint).

Usage:
    from app.parent.coder_v2 import DeepSeekCoderV2
    coder = DeepSeekCoderV2()
    text = coder.complete("...prompt...")
    json_text = coder.complete_json("...prompt that returns JSON...")

Environment:
    NERION_CODER_MODEL     – model name (default: 'deepseek-coder-v2')
    NERION_CODER_BASE_URL  – optional base URL for Ollama (e.g., http://localhost:11434)

This module is resilient: if langchain_ollama is unavailable, calls return None
so callers can gracefully fall back to heuristic planners.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any


class DeepSeekCoderV2:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, temperature: float = 0.1) -> None:
        raw = (model or os.getenv("NERION_CODER_MODEL") or "deepseek-coder-v2").strip()
        # Normalize common provider prefixes like "ollama:deepseek-coder-v2"
        self.model = (raw.split(":", 1)[-1] if ":" in raw else raw).strip()
        self.base_url = (base_url or os.getenv("NERION_CODER_BASE_URL") or "").strip() or None
        self.temperature = float(temperature)

    def _build_client(self, *, json_mode: bool = False):
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception:
            return None
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if json_mode:
            # Hint JSON output mode when supported
            kwargs["format"] = "json"
        try:
            return ChatOllama(**kwargs)
        except Exception:
            return None

    def complete(self, prompt: str, system: Optional[str] = None) -> Optional[str]:
        """Return raw string completion from the coder model (or None on failure)."""
        client = self._build_client(json_mode=False)
        if client is None:
            return None
        try:
            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "human", "content": prompt})
            resp = client.invoke(messages)
            return getattr(resp, "content", None) or str(resp)
        except Exception:
            return None

    def complete_json(self, prompt: str, system: Optional[str] = None) -> Optional[str]:
        """Return JSON‑formatted completion (string) or None on failure.

        Uses json format hint where supported and falls back to plain mode.
        """
        client = self._build_client(json_mode=True) or self._build_client(json_mode=False)
        if client is None:
            return None
        try:
            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "human", "content": prompt})
            resp = client.invoke(messages)
            return getattr(resp, "content", None) or str(resp)
        except Exception:
            return None

__all__ = ["DeepSeekCoderV2"]
