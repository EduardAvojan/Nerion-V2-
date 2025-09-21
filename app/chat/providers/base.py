"""Provider registry and adapters for Nerion V2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import time

import requests
import yaml

CONFIG_PATH = os.getenv("NERION_MODEL_CATALOG", "config/model_catalog.yaml")
SETTINGS_PATH = os.getenv("NERION_SETTINGS_PATH", "app/settings.yaml")
DEFAULT_TIMEOUT = float(os.getenv("NERION_V2_REQUEST_TIMEOUT", "15"))


class ProviderError(RuntimeError):
    """Generic provider failure."""


class ProviderNotConfigured(ProviderError):
    """Raised when a provider is missing credentials or configuration."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected YAML root type in {path!r}")
        return data
    except FileNotFoundError:
        return {}


class _ProviderAdapter:
    """Abstract provider adapter."""

    name: str

    def __init__(self, name: str, endpoint: str, api_key: Optional[str]):
        self.name = name
        self.endpoint = (endpoint or "").rstrip("/")
        self.api_key = (api_key or "").strip() or None

    def ensure_ready(self, model: str) -> None:
        if not self.endpoint:
            raise ProviderNotConfigured(f"Provider '{self.name}' missing endpoint")
        if not self.api_key:
            raise ProviderNotConfigured(f"Provider '{self.name}' missing API key")
        if not model:
            raise ProviderNotConfigured(f"Provider '{self.name}' missing model identifier")

    def generate(
        self,
        *,
        model: str,
        prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
        messages: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        raise NotImplementedError


_OPENAI_TEMPERATURE_ALWAYS_DEFAULT = {
    "o4-mini",
    "gpt-4o-mini",
}


class _OpenAIAdapter(_ProviderAdapter):
    def generate(
        self,
        *,
        model: str,
        prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
        messages: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        self.ensure_ready(model)
        url = f"{self.endpoint}/chat/completions"
        payload: Dict[str, Any] = {"model": model}
        temp_val = max(0.0, min(2.0, float(temperature)))
        if model not in _OPENAI_TEMPERATURE_ALWAYS_DEFAULT:
            payload["temperature"] = temp_val
        def _coerce_messages(raw: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
            formatted: List[Dict[str, str]] = []
            if not raw:
                return formatted
            for item in raw:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").lower()
                if role not in {"system", "user", "assistant"}:
                    role = "user"
                formatted.append({"role": role, "content": item.get("content") or ""})
            return formatted

        formatted_messages = _coerce_messages(messages)
        if formatted_messages:
            payload["messages"] = formatted_messages
        else:
            sys_prompt = "You are Nerion, a helpful assistant."
            payload["messages"] = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt or ""},
            ]
        if max_tokens is not None:
            payload["max_completion_tokens"] = int(max_tokens)
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:  # pragma: no cover - network failures
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        latency = time.perf_counter() - started
        if response.status_code >= 400:
            raise ProviderError(
                f"OpenAI error {response.status_code}: {response.text.strip()}"
            )
        data = response.json()
        message = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        usage = data.get("usage") or {}
        return LLMResponse(
            text=message.strip(),
            provider="openai",
            model=model,
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class _AnthropicAdapter(_ProviderAdapter):
    API_VERSION = "2023-06-01"

    def generate(
        self,
        *,
        model: str,
        prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
        messages: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        self.ensure_ready(model)
        url = f"{self.endpoint}/messages"
        formatted: List[Dict[str, str]] = []
        system_prompt: Optional[str] = None
        if messages:
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").lower()
                content = item.get("content") or ""
                if role == "system":
                    system_prompt = content
                elif role in {"user", "assistant"}:
                    formatted.append({"role": role, "content": content})
        if not formatted:
            formatted = [{"role": "user", "content": prompt or ""}]
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens or 1024),
            "messages": formatted,
            "temperature": max(0.0, min(1.0, float(temperature))),
        }
        if system_prompt:
            payload["system"] = system_prompt
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        started = time.perf_counter()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:  # pragma: no cover
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
        latency = time.perf_counter() - started
        if response.status_code >= 400:
            raise ProviderError(
                f"Anthropic error {response.status_code}: {response.text.strip()}"
            )
        data = response.json()
        content = data.get("content") or []
        text = "\n".join([part.get("text", "") for part in content if isinstance(part, dict)])
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text.strip(),
            provider="anthropic",
            model=model,
            latency_s=latency,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )


class _GoogleAdapter(_ProviderAdapter):
    def generate(
        self,
        *,
        model: str,
        prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
        messages: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        self.ensure_ready(model)
        url = f"{self.endpoint}/models/{model}:generateContent"

        system_instruction: Optional[Dict[str, Any]] = None
        contents: List[Dict[str, Any]] = []

        def _parts(text: str) -> List[Dict[str, str]]:
            return [{"text": text}]

        if messages:
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").lower()
                text = item.get("content") or ""
                if role == "system":
                    system_instruction = {"parts": _parts(text)}
                else:
                    contents.append({
                        "role": "user" if role != "assistant" else "model",
                        "parts": _parts(text),
                    })
        if not contents:
            contents.append({
                "role": "user",
                "parts": _parts(prompt or ""),
            })

        generation_cfg: Dict[str, Any] = {
            "temperature": max(0.0, min(2.0, float(temperature))),
        }
        if max_tokens is not None:
            generation_cfg["maxOutputTokens"] = int(max_tokens)
        if response_format == "json_object":
            generation_cfg["responseMimeType"] = "application/json"

        payload: Dict[str, Any] = {"contents": contents, "generationConfig": generation_cfg}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        started = time.perf_counter()
        try:
            response = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network failures
            raise ProviderError(f"Gemini request failed: {exc}") from exc
        latency = time.perf_counter() - started
        if response.status_code >= 400:
            raise ProviderError(
                f"Gemini error {response.status_code}: {response.text.strip()}"
            )
        data = response.json()
        candidates = data.get("candidates") or []
        parts_out: List[str] = []
        for cand in candidates:
            content = cand.get("content") or {}
            for part in content.get("parts", []) or []:
                fragment = part.get("text")
                if fragment:
                    parts_out.append(str(fragment))
        combined = "\n".join(parts_out).strip()
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=combined,
            provider="google",
            model=model,
            latency_s=latency,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )


_ADAPTERS = {
    "openai": _OpenAIAdapter,
    "anthropic": _AnthropicAdapter,
    "google": _GoogleAdapter,
}


class ProviderRegistry:
    """Loads configuration and hands out provider adapters."""

    def __init__(self, catalog: Dict[str, Any], settings: Dict[str, Any]):
        self._catalog = catalog
        self._settings = settings or {}
        api_section = catalog.get("api_providers") or {}
        self._defaults: Dict[str, str] = api_section.get("defaults", {})
        self._providers: Dict[str, Dict[str, Any]] = api_section.get("providers", {})
        self._cache: Dict[Tuple[str, str], _ProviderAdapter] = {}

    @classmethod
    def from_files(
        cls,
        catalog_path: str = CONFIG_PATH,
        settings_path: str = SETTINGS_PATH,
    ) -> "ProviderRegistry":
        return cls(_load_yaml(catalog_path), _load_yaml(settings_path))

    def _resolve_provider_id(self, role: str, *, use_env: bool = True) -> Optional[str]:
        role_key = (role or "chat").upper().replace("-", "_")
        if use_env:
            env_key = os.getenv(f"NERION_V2_{role_key}_PROVIDER")
            if env_key:
                return env_key
        llm_settings = (self._settings.get("llm") or {})
        role_overrides = llm_settings.get("roles") or {}
        if role in role_overrides:
            return role_overrides[role]
        if role == "chat":
            return llm_settings.get("default_provider") or self._defaults.get("chat")
        if role == "planner":
            return llm_settings.get("fallback_provider") or self._defaults.get("planner")
        if role == "code":
            return self._defaults.get("code") or llm_settings.get("default_provider")
        if role == "embeddings":
            return self._defaults.get("embeddings")
        return llm_settings.get("default_provider") or self._defaults.get("chat")

    @staticmethod
    def _split_provider_id(provider_id: str) -> Tuple[str, str]:
        if not provider_id or ":" not in provider_id:
            raise ProviderNotConfigured(
                "Provider identifier must look like 'provider:model'"
            )
        provider, model = provider_id.split(":", 1)
        return provider.strip(), model.strip()

    def _get_provider_spec(self, provider: str) -> Dict[str, Any]:
        spec = self._providers.get(provider)
        if not spec:
            raise ProviderNotConfigured(f"Unknown provider '{provider}'")
        return spec

    def _get_model_spec(self, provider: str, model: str) -> Dict[str, Any]:
        spec = self._get_provider_spec(provider)
        models = spec.get("models") or {}
        if model not in models:
            raise ProviderNotConfigured(
                f"Model '{model}' not configured for provider '{provider}'"
            )
        return models[model]

    def resolve(
        self,
        role: str = "chat",
        provider_override: Optional[str] = None,
    ) -> Tuple[_ProviderAdapter, str, Dict[str, Any]]:
        override = (provider_override or "").strip()
        provider_id = override or self._resolve_provider_id(role)
        if not provider_id:
            raise ProviderNotConfigured(
                f"No provider configured for role '{role}'"
            )
        provider_name, model_name = self._split_provider_id(provider_id)
        provider_spec = self._get_provider_spec(provider_name)
        model_spec = self._get_model_spec(provider_name, model_name)
        cache_key = (provider_name, provider_spec.get("endpoint", ""))
        adapter = self._cache.get(cache_key)
        if adapter is None:
            adapter_cls = _ADAPTERS.get(provider_name)
            if adapter_cls is None:
                raise ProviderNotConfigured(
                    f"Provider '{provider_name}' is not supported yet"
                )
            api_key = os.getenv(provider_spec.get("key_env", ""))
            adapter = adapter_cls(
                provider_name,
                provider_spec.get("endpoint", ""),
                api_key,
            )
            self._cache[cache_key] = adapter
        return adapter, model_name, model_spec

    def generate(
        self,
        *,
        role: str,
        prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        response_format: Optional[str] = None,
        provider_override: Optional[str] = None,
    ) -> LLMResponse:
        adapter, model_name, model_spec = self.resolve(role, provider_override=provider_override)
        timeout = float(
            os.getenv("NERION_V2_REQUEST_TIMEOUT")
            or (self._settings.get("llm") or {}).get("request_timeout_seconds")
            or DEFAULT_TIMEOUT
        )
        max_toks = max_tokens or model_spec.get("max_output_tokens")
        return adapter.generate(
            model=model_name,
            prompt=prompt or "",
            temperature=temperature,
            max_tokens=max_toks,
            timeout=timeout,
            messages=messages,
            response_format=response_format,
        )

    def active_provider(self, role: str) -> Optional[str]:
        return self._resolve_provider_id(role, use_env=True)

    def default_provider(self, role: str) -> Optional[str]:
        return self._resolve_provider_id(role, use_env=False)

    def list_role_options(self) -> Dict[str, List[Dict[str, Any]]]:
        options: Dict[str, List[Dict[str, Any]]] = {}
        for provider_name, provider_spec in self._providers.items():
            models = provider_spec.get("models") or {}
            for model_name, meta in models.items():
                provider_id = f"{provider_name}:{model_name}"
                roles = list(meta.get("roles") or [])
                if not roles:
                    roles = ["chat"]
                label = meta.get("label")
                if not label:
                    provider_title = provider_name.replace('_', ' ').replace('-', ' ').title()
                    label = f"{provider_title} · {model_name}"
                note_parts: List[str] = []
                if meta.get("supports_structured_output"):
                    note_parts.append("json")
                if meta.get("supports_tools"):
                    note_parts.append("tools")
                if meta.get("multimodal"):
                    note_parts.append("multimodal")
                note = ", ".join(note_parts)
                entry = {
                    "provider_id": provider_id,
                    "provider": provider_name,
                    "model": model_name,
                    "label": label,
                    "note": note,
                    "roles": roles,
                }
                for role in roles:
                    options.setdefault(role, []).append(entry)
        for role, entries in options.items():
            entries.sort(key=lambda item: item.get("label") or item.get("provider_id"))
        return options


_REGISTRY: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ProviderRegistry.from_files()
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
