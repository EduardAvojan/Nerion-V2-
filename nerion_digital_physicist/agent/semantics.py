"""Semantic embedding utilities for Phase 4 world models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.chat.providers.base import (
    ProviderError,
    ProviderNotConfigured,
    ProviderRegistry,
    DEFAULT_TIMEOUT,
)

DEFAULT_DIMENSION = 16
_CACHE_FILENAME = ".semantic_cache.json"


class SemanticEmbedder:
    """Generate deterministic semantic embeddings for code snippets.

    The embedder is designed to be LLM pluggable. When no external provider is
    available, it falls back to a deterministic hash-based embedding so the
    learning loop can proceed during local development and testing.
    """

    def __init__(
        self,
        *,
        dimension: int = DEFAULT_DIMENSION,
        cache_path: Optional[Path] = None,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.cache_path = cache_path or Path(__file__).resolve().parent / _CACHE_FILENAME
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = {}
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}

        timeout_value: Optional[float] = None
        if timeout is not None:
            timeout_value = float(timeout)
        else:
            timeout_env = os.getenv("NERION_SEMANTIC_TIMEOUT")
            if timeout_env:
                try:
                    timeout_value = float(timeout_env)
                except ValueError:
                    timeout_value = None
        if not timeout_value or timeout_value <= 0:
            timeout_value = DEFAULT_TIMEOUT
        self.timeout = float(timeout_value)
        provider_id = provider or os.getenv("NERION_SEMANTIC_PROVIDER", "hash")
        provider_id = provider_id.strip().lower()
        self._registry: Optional[ProviderRegistry] = None
        self._provider_override: Optional[str] = None
        self.dimension = dimension
        if provider_id and provider_id != "hash":
            self._initialise_registry(provider_id)
        else:
            self.provider = "hash"
            self._provider_override = None

    def _initialise_registry(self, provider_id: str) -> None:
        self.provider = provider_id
        self._registry = ProviderRegistry.from_files()
        try:
            adapter, model_name, model_spec = self._registry.resolve(
                "embeddings", provider_override=provider_id
            )
        except (ProviderError, ProviderNotConfigured):
            self.provider = "hash"
            self._registry = None
            self._provider_override = None
            self.dimension = max(1, self.dimension)
            return
        self._provider_override = f"{adapter.name}:{model_name}"
        try:
            dimensionality = int(model_spec.get("dimensionality") or self.dimension)
            if dimensionality > 0:
                self.dimension = dimensionality
        except Exception:
            pass

    def embed(self, identifier: str, text: str) -> List[float]:
        """Return a semantic vector for the supplied code snippet."""

        cache_key = self._build_cache_key(identifier, text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.provider == "hash" or not self._registry or not self._provider_override:
            vector = self._hash_embedding(text)
        else:
            vector = self._provider_embedding(text)

        self._cache[cache_key] = vector
        self._persist_cache()
        return vector

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_cache_key(self, identifier: str, text: str) -> str:
        digest = hashlib.sha256(f"{identifier}::{text}".encode("utf-8")).hexdigest()
        return f"{self.dimension}:{self.provider}:{digest}"

    def _hash_embedding(self, text: str) -> List[float]:
        seed = text.encode("utf-8")
        vector: List[float] = []
        for idx in range(self.dimension):
            hasher = hashlib.sha256()
            hasher.update(seed)
            hasher.update(idx.to_bytes(4, "little", signed=False))
            digest = hasher.digest()
            integer = int.from_bytes(digest[:8], "big", signed=False)
            value = (integer / (2**64 - 1)) * 2.0 - 1.0
            vector.append(float(value))
        return vector

    def _provider_embedding(self, text: str) -> List[float]:
        assert self._registry is not None
        assert self._provider_override is not None
        try:
            vectors = self._registry.embed(
                [text],
                provider_override=self._provider_override,
                timeout=self.timeout,
            )
        except ProviderError:
            return self._hash_embedding(text)
        if not vectors:
            return self._hash_embedding(text)
        vector = vectors[0]
        if len(vector) != self.dimension:
            try:
                self.dimension = len(vector)
            except Exception:
                pass
        return [float(val) for val in vector]

    def _persist_cache(self) -> None:
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8"
        )


_GLOBAL_EMBEDDER: Optional[SemanticEmbedder] = None


def get_global_embedder() -> SemanticEmbedder:
    """Return a process-wide embedder instance."""

    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = SemanticEmbedder()
    return _GLOBAL_EMBEDDER
