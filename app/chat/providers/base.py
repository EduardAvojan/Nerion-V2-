"""Provider registry and adapters for Nerion V2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import time

import requests
import yaml

try:
    from ops.telemetry.events import (
        record_completion as _telemetry_record_completion,
        record_prompt as _telemetry_record_prompt,
    )
except Exception:  # pragma: no cover - telemetry optional at runtime
    def _telemetry_record_prompt(*_args, **_kwargs):  # type: ignore
        return None

    def _telemetry_record_completion(*_args, **_kwargs):  # type: ignore
        return None

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

    def embed(
        self,
        *,
        model: str,
        texts: List[str],
        timeout: float,
    ) -> List[List[float]]:
        """Compute embeddings for ``texts`` using the provider model."""
        raise NotImplementedError


_OPENAI_TEMPERATURE_ALWAYS_DEFAULT = {
    "gpt-4o-mini",
    "gpt-5",
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

    def embed(
        self,
        *,
        model: str,
        texts: List[str],
        timeout: float,
    ) -> List[List[float]]:
        self.ensure_ready(model)
        inputs = [str(text) for text in texts if text is not None]
        if not inputs:
            return []
        url = f"{self.endpoint}/embeddings"
        payload: Dict[str, Any] = {"model": model, "input": inputs}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:  # pragma: no cover - network failures
            raise ProviderError(f"OpenAI embeddings request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"OpenAI embeddings error {response.status_code}: {response.text.strip()}"
            )
        data = response.json()
        items = data.get("data") or []
        embeddings: List[List[float]] = []
        for item in items:
            vector = item.get("embedding")
            if isinstance(vector, list):
                try:
                    embeddings.append([float(val) for val in vector])
                except Exception as exc:
                    raise ProviderError("OpenAI embedding vector contains non-numeric values") from exc
        if len(embeddings) != len(inputs):
            raise ProviderError("OpenAI embeddings response missing entries for some inputs")
        return embeddings


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

    def embed(
        self,
        *,
        model: str,
        texts: List[str],
        timeout: float,
    ) -> List[List[float]]:
        self.ensure_ready(model)
        inputs = [str(text) for text in texts if text is not None]
        if not inputs:
            return []
        url = f"{self.endpoint}/models/{model}:batchEmbedContents"
        payload: Dict[str, Any] = {
            "requests": [
                {"content": {"parts": [{"text": text}]}}
                for text in inputs
            ]
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network failures
            raise ProviderError(f"Gemini embeddings request failed: {exc}") from exc
        latency = time.perf_counter() - started
        if response.status_code >= 400:
            raise ProviderError(
                f"Gemini embeddings error {response.status_code}: {response.text.strip()}"
            )
        data = response.json()
        embeddings_raw = data.get("embeddings") or []
        embeddings: List[List[float]] = []
        for item in embeddings_raw:
            values = item.get("values") or []
            if not isinstance(values, list):
                raise ProviderError("Gemini embeddings response malformed")
            try:
                embeddings.append([float(val) for val in values])
            except Exception as exc:
                raise ProviderError("Gemini embedding vector contains non-numeric values") from exc
        if len(embeddings) != len(inputs):
            raise ProviderError("Gemini embeddings response count mismatch")
        # Latency currently unused but captured for parity with generation path.
        _ = latency
        return embeddings


class _AnthropicAdapter(_ProviderAdapter):
    """Adapter for Anthropic Claude API."""

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

        # Build messages in Anthropic format
        anthropic_messages: List[Dict[str, str]] = []
        system_prompt: Optional[str] = None

        if messages:
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").lower()
                content = item.get("content") or ""

                if role == "system":
                    system_prompt = content
                elif role in ("user", "assistant"):
                    anthropic_messages.append({
                        "role": role,
                        "content": content
                    })
                else:
                    # Default unknown roles to user
                    anthropic_messages.append({
                        "role": "user",
                        "content": content
                    })

        if not anthropic_messages:
            anthropic_messages.append({
                "role": "user",
                "content": prompt or ""
            })

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": int(max_tokens) if max_tokens else 4096,
            "temperature": max(0.0, min(1.0, float(temperature))),
        }

        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt

        # Headers for Anthropic API
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Log API call for debugging
        print(f"[API] Calling Anthropic API: {model} (url: {url})")

        started = time.perf_counter()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:  # pragma: no cover - network failures
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
        latency = time.perf_counter() - started

        print(f"[API] Anthropic response received in {latency:.2f}s")

        if response.status_code >= 400:
            raise ProviderError(
                f"Anthropic error {response.status_code}: {response.text.strip()}"
            )

        data = response.json()

        # Extract text from response
        content_blocks = data.get("content") or []
        text_parts: List[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        combined_text = "\n".join(text_parts).strip()

        # Extract token usage
        usage = data.get("usage") or {}

        return LLMResponse(
            text=combined_text,
            provider="anthropic",
            model=model,
            latency_s=latency,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )

    def embed(
        self,
        *,
        model: str,
        texts: List[str],
        timeout: float,
    ) -> List[List[float]]:
        # Anthropic doesn't provide embeddings API yet
        raise ProviderError("Anthropic does not support embeddings")


class _VertexAIAdapter(_ProviderAdapter):
    """Adapter for Google Cloud Vertex AI generative models."""

    def __init__(self, name: str, endpoint: str, api_key: Optional[str]):
        super().__init__(name, endpoint, api_key)
        self._project_id: Optional[str] = None
        self._location: Optional[str] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize Vertex AI SDK with project and location."""
        if self._initialized:
            return

        import vertexai

        # Extract project_id and location from endpoint or environment
        # Endpoint format: "projects/PROJECT_ID/locations/LOCATION"
        if self.endpoint:
            parts = self.endpoint.split("/")
            if len(parts) >= 4 and parts[0] == "projects" and parts[2] == "locations":
                self._project_id = parts[1]
                self._location = parts[3]

        # Fall back to environment variables if not in endpoint
        if not self._project_id:
            self._project_id = os.getenv("NERION_V2_VERTEX_PROJECT_ID")
        if not self._location:
            self._location = os.getenv("NERION_V2_VERTEX_LOCATION", "us-central1")

        if not self._project_id:
            raise ProviderNotConfigured(
                "Vertex AI requires project_id in endpoint or NERION_V2_VERTEX_PROJECT_ID env var"
            )

        # Initialize Vertex AI
        # Only set credentials path if explicitly provided AND the file exists
        # Vertex AI Custom Jobs use automatic service account authentication
        credentials_path = os.getenv("NERION_V2_VERTEX_CREDENTIALS")
        if credentials_path and os.path.isfile(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        vertexai.init(project=self._project_id, location=self._location)
        self._initialized = True

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
        self._ensure_initialized()

        from vertexai.generative_models import GenerativeModel

        # Create model instance - just use the model name directly
        vertex_model = GenerativeModel(model)

        # Build messages in Vertex AI format
        vertex_messages = []
        system_instruction = None

        if messages:
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").lower()
                text = item.get("content") or ""

                if role == "system":
                    system_instruction = text
                else:
                    # Vertex AI uses "user" and "model" roles
                    vertex_role = "user" if role != "assistant" else "model"
                    vertex_messages.append({
                        "role": vertex_role,
                        "parts": [{"text": text}]
                    })

        if not vertex_messages:
            vertex_messages.append({
                "role": "user",
                "parts": [{"text": prompt or ""}]
            })

        # Build generation config
        generation_config = {
            "temperature": max(0.0, min(2.0, float(temperature))),
        }

        if max_tokens is not None:
            generation_config["max_output_tokens"] = int(max_tokens)

        if response_format == "json_object":
            generation_config["response_mime_type"] = "application/json"

        # Add system instruction if provided
        if system_instruction:
            # Prepend system instruction to first user message
            if vertex_messages and vertex_messages[0]["role"] == "user":
                original_text = vertex_messages[0]["parts"][0]["text"]
                vertex_messages[0]["parts"][0]["text"] = f"{system_instruction}\n\n{original_text}"

        started = time.perf_counter()
        try:
            # Build content for Vertex AI - it expects a list of Content objects or simple text
            if len(vertex_messages) == 1 and vertex_messages[0]["role"] == "user":
                # Simple single message case - just pass the text
                content = vertex_messages[0]["parts"][0]["text"]
            else:
                # Multi-turn conversation
                content = vertex_messages

            response = vertex_model.generate_content(
                content,
                generation_config=generation_config
            )
            latency = time.perf_counter() - started

            # Extract text from response
            text = response.text if hasattr(response, 'text') else ""

            # Extract token usage if available
            prompt_tokens = None
            completion_tokens = None
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', None)
                completion_tokens = getattr(usage, 'candidates_token_count', None)

            return LLMResponse(
                text=text.strip(),
                provider="vertexai",
                model=model,
                latency_s=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"Vertex AI generation failed: {exc}") from exc

    def embed(
        self,
        *,
        model: str,
        texts: List[str],
        timeout: float,
    ) -> List[List[float]]:
        self._ensure_initialized()

        from vertexai.language_models import TextEmbeddingModel

        inputs = [str(text) for text in texts if text is not None]
        if not inputs:
            return []

        try:
            embedding_model = TextEmbeddingModel.from_pretrained(model)
            embeddings_response = embedding_model.get_embeddings(inputs)

            embeddings: List[List[float]] = []
            for emb in embeddings_response:
                embeddings.append(emb.values)

            if len(embeddings) != len(inputs):
                raise ProviderError("Vertex AI embeddings response count mismatch")

            return embeddings
        except Exception as exc:
            raise ProviderError(f"Vertex AI embeddings failed: {exc}") from exc


_ADAPTERS = {
    "openai": _OpenAIAdapter,
    "google": _GoogleAdapter,
    "anthropic": _AnthropicAdapter,
    "vertexai": _VertexAIAdapter,
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
        # Priority 1: UI settings (for mid-session switching via Mission Control)
        # Read on EVERY request to allow real-time provider changes
        from app.chat.ui_settings import get_provider_for_role
        ui_provider = get_provider_for_role(role)
        if ui_provider:
            print(f"[PROVIDER] Resolved role '{role}' from UI settings: {ui_provider}")
            return ui_provider

        # Priority 2: Environment variables (for CLI/script overrides)
        role_key = (role or "chat").upper().replace("-", "_")
        if use_env:
            env_key = os.getenv(f"NERION_V2_{role_key}_PROVIDER")
            if env_key:
                print(f"[PROVIDER] Resolved role '{role}' from env NERION_V2_{role_key}_PROVIDER: {env_key}")
                return env_key

        # Priority 3: settings.yaml role overrides
        llm_settings = (self._settings.get("llm") or {})
        role_overrides = llm_settings.get("roles") or {}
        if role in role_overrides:
            result = role_overrides[role]
            print(f"[PROVIDER] Resolved role '{role}' from settings.llm.roles: {result}")
            return result

        # Priority 4: Fallback to model_catalog.yaml defaults
        if role == "chat":
            result = llm_settings.get("default_provider") or self._defaults.get("chat")
            print(f"[PROVIDER] Resolved role '{role}' from defaults.chat: {result}")
            return result
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

        subject = f"{adapter.name}:{model_name}"
        base_metadata = {
            "role": role,
            "provider": adapter.name,
            "model": model_name,
            "temperature": float(temperature),
            "timeout_s": float(timeout),
        }
        if max_toks is not None:
            try:
                base_metadata["max_tokens"] = int(max_toks)
            except Exception:
                pass
        if response_format:
            base_metadata["response_format"] = response_format

        try:
            prompt_kwargs = {
                "source": f"app.chat.providers.{adapter.name}",
                "subject": subject,
                "metadata": dict(base_metadata),
                "tags": ["provider", role],
            }
            if messages:
                prompt_kwargs["messages"] = messages
            else:
                prompt_kwargs["prompt"] = prompt or ""
            _telemetry_record_prompt(**prompt_kwargs)
        except Exception:  # pragma: no cover - do not break provider flow
            pass

        result = adapter.generate(
            model=model_name,
            prompt=prompt or "",
            temperature=temperature,
            max_tokens=max_toks,
            timeout=timeout,
            messages=messages,
            response_format=response_format,
        )
        try:
            completion_meta = dict(base_metadata)
            completion_meta.update(
                {
                    "latency_ms": int(result.latency_s * 1000),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost_usd": result.cost_usd,
                }
            )
            _telemetry_record_completion(
                source=f"app.chat.providers.{adapter.name}",
                text=result.text,
                subject=subject,
                metadata=completion_meta,
                tags=["provider", role],
            )
        except Exception:  # pragma: no cover - telemetry must not break flow
            pass
        return result

    def embed(
        self,
        texts: List[str],
        *,
        provider_override: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> List[List[float]]:
        adapter, model_name, _model_spec = self.resolve("embeddings", provider_override=provider_override)
        request_timeout = float(
            os.getenv("NERION_V2_REQUEST_TIMEOUT")
            or (self._settings.get("llm") or {}).get("request_timeout_seconds")
            or DEFAULT_TIMEOUT
        )
        embed_timeout = float(timeout or request_timeout)
        return adapter.embed(model=model_name, texts=list(texts), timeout=embed_timeout)

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
