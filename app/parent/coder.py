"""API-backed coder helper for Nerion V2."""
from __future__ import annotations

from typing import Optional, Dict, List
import warnings

from app.chat.providers import (
    ProviderError,
    ProviderNotConfigured,
    get_registry,
)


class Coder:
    """Thin wrapper around the provider registry for code-oriented prompts."""

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        role: str = "code",
        provider_override: Optional[str] = None,
    ) -> None:
        if model or backend or base_url:
            warnings.warn(
                "Coder no longer accepts local backend parameters; configured providers in app/settings.yaml are used instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.temperature = float(temperature)
        self.role = role
        self.provider_override = provider_override
        self._registry = get_registry()

    @staticmethod
    def _messages(prompt: str, system: Optional[str]) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _generate(
        self,
        prompt: str,
        *,
        system: Optional[str],
        response_format: Optional[str] = None,
    ) -> Optional[str]:
        messages = self._messages(prompt, system)
        if response_format == "json_object":
            has_json_hint = any(
                msg.get("role") == "system" and "json" in (msg.get("content") or "").lower()
                for msg in messages
            )
            if not has_json_hint:
                messages = ([{"role": "system", "content": "Respond with a valid JSON object."}]
                    + messages)
        try:
            resp = self._registry.generate(
                role=self.role,
                messages=messages,
                temperature=self.temperature,
                response_format=response_format,
                provider_override=self.provider_override,
            )
        except ProviderNotConfigured:
            return None
        except ProviderError as exc:
            warnings.warn(f"Coder provider error: {exc}", RuntimeWarning, stacklevel=2)
            return None
        return resp.text or None

    def complete(self, prompt: str, system: Optional[str] = None) -> Optional[str]:
        return self._generate(prompt, system=system)

    def complete_json(self, prompt: str, system: Optional[str] = None) -> Optional[str]:
        return self._generate(prompt, system=system, response_format="json_object")


__all__ = ["Coder"]
