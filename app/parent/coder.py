"""Unified local coder interface with multiple backends.

Backends (env NERION_CODER_BACKEND):
  - ollama (default): uses langchain_ollama.ChatOllama
  - llama_cpp: uses llama_cpp.Llama (GGUF); requires LLAMA_CPP_MODEL_PATH
  - vllm: OpenAI-compatible HTTP endpoint at NERION_CODER_BASE_URL
  - exllamav2: optional exllamav2 bindings (requires EXLLAMA_MODEL_DIR)

Models (NERION_CODER_MODEL):
  - e.g., deepseek-coder-v2, qwen2.5-coder, starcoder2, codellama (quantized)

All imports are optional; if unavailable, calls return None.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import os


class Coder:
    def __init__(self, model: Optional[str] = None, backend: Optional[str] = None, base_url: Optional[str] = None, temperature: float = 0.1) -> None:
        raw_backend = (backend or os.getenv("NERION_CODER_BACKEND") or "ollama").strip().lower()
        self.backend = raw_backend
        raw_model = (model or os.getenv("NERION_CODER_MODEL") or "deepseek-coder-v2").strip()
        # Normalize prefixed forms like "ollama:deepseek-coder-v2"
        if ":" in raw_model and raw_model.split(":", 1)[0] in {"ollama", "llama_cpp", "vllm", "exllamav2"}:
            raw_model = raw_model.split(":", 1)[1]
        self.model = raw_model
        self.base_url = (base_url or os.getenv("NERION_CODER_BASE_URL") or "").strip() or None
        self.temperature = float(temperature)

    # -------- public API --------
    def complete(self, prompt: str, system: Optional[str] = None) -> Optional[str]:
        client = self._build_client(json_mode=False)
        if client is None:
            return None
        try:
            if self.backend == "ollama":
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "human", "content": prompt})
                resp = client.invoke(messages)
                return getattr(resp, "content", None) or str(resp)
            elif self.backend == "llama_cpp":
                # Basic completion interface
                out = client(prompt, max_tokens=256, temperature=self.temperature)
                txt = out.get("choices", [{}])[0].get("text") if isinstance(out, dict) else None
                return txt or None
            elif self.backend == "vllm":
                # OpenAI-compatible /v1/completions
                import requests  # type: ignore
                url = (self.base_url or "").rstrip("/") + "/v1/completions"
                js = {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": self.temperature,
                }
                r = requests.post(url, json=js, timeout=15)
                if r.ok:
                    data = r.json()
                    return (data.get("choices") or [{}])[0].get("text")
                return None
            elif self.backend == "exllamav2":
                # Minimal exllamav2 loop
                gen = client
                out = gen.generate(prompt, max_new_tokens=256, temperature=self.temperature)  # type: ignore
                return str(out)
        except Exception:
            return None
        return None

    def complete_json(self, prompt: str, system: Optional[str] = None) -> Optional[str]:
        client = self._build_client(json_mode=True)
        if client is None:
            return None
        try:
            if self.backend == "ollama":
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "human", "content": prompt})
                resp = client.invoke(messages)
                return getattr(resp, "content", None) or str(resp)
            # Other backends: fall back to plain completion
            return self.complete(prompt, system)
        except Exception:
            return None

    # -------- clients --------
    def _build_client(self, *, json_mode: bool):
        try:
            if self.backend == "ollama":
                from langchain_ollama import ChatOllama  # type: ignore
                kwargs: Dict[str, Any] = {"model": self.model, "temperature": self.temperature}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                if json_mode:
                    kwargs["format"] = "json"
                return ChatOllama(**kwargs)
            if self.backend == "llama_cpp":
                from llama_cpp import Llama  # type: ignore
                model_path = os.getenv("LLAMA_CPP_MODEL_PATH")
                if not model_path:
                    return None
                return Llama(model_path=model_path)
            if self.backend == "vllm":
                # HTTP client only; ensure base_url exists
                return object() if (self.base_url or os.getenv("NERION_CODER_BASE_URL")) else None
            if self.backend == "exllamav2":
                # Experimental: try to import exllamav2 if available; otherwise emit a helpful warning.
                model_dir = os.getenv("EXLLAMA_MODEL_DIR")
                if not model_dir:
                    print("[coder:exllamav2] EXLLAMA_MODEL_DIR not set; see docs/models.md (experimental backend)")
                    return None
                try:
                    # Attempt to import a common API; different releases expose slightly different names.
                    # We return a tiny adapter with a 'generate(prompt,...)' method when possible.
                    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer  # type: ignore
                    from exllamav2.generator import ExLlamaV2Generator  # type: ignore
                    cfg = ExLlamaV2Config(model_dir)
                    model = ExLlamaV2(cfg)
                    tokenizer = ExLlamaV2Tokenizer(cfg)
                    generator = ExLlamaV2Generator(model, tokenizer)
                    class _Gen:
                        def generate(self, prompt, max_new_tokens=256, temperature=0.1):
                            try:
                                return generator.generate_simple(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                            except Exception:
                                return None
                    return _Gen()
                except Exception:
                    # If the expected API isn't present, keep behavior graceful with a clear notice.
                    print("[coder:exllamav2] Python bindings not available or incompatible; backend is experimental.")
                    return None
        except Exception:
            return None
        return None
