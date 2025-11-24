"""Semantic embedding utilities for Phase 4 world models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from app.chat.providers.base import (
    ProviderError,
    ProviderNotConfigured,
    ProviderRegistry,
    DEFAULT_TIMEOUT,
)

DEFAULT_DIMENSION = 16
_CACHE_FILENAME = ".semantic_cache.json"

# CodeBERT support
_CODEBERT_MODEL = None
_CODEBERT_TOKENIZER = None
CODEBERT_DIM = 768

# GraphCodeBERT model support (for on-the-fly embedding generation)
_GRAPHCODEBERT_MODEL = None
_GRAPHCODEBERT_TOKENIZER = None
GRAPHCODEBERT_DIM = 768

# GraphCodeBERT pre-computed embeddings support (for curriculum lessons)
_GRAPHCODEBERT_EMBEDDINGS = None
_GRAPHCODEBERT_LESSON_MAP = None


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
        # Default to GraphCodeBERT for 768-dim embeddings (required by trained 91.8% GNN model)
        # Can be overridden via NERION_SEMANTIC_PROVIDER environment variable
        provider_id = provider or os.getenv("NERION_SEMANTIC_PROVIDER", "graphcodebert")
        provider_id = provider_id.strip().lower()
        self._registry: Optional[ProviderRegistry] = None
        self._provider_override: Optional[str] = None
        self.dimension = dimension

        # Check for GraphCodeBERT provider (pre-computed embeddings)
        if provider_id == "graphcodebert":
            self.provider = "graphcodebert"
            self._provider_override = None
            self.dimension = GRAPHCODEBERT_DIM
        # Check for CodeBERT provider
        elif provider_id == "codebert":
            self.provider = "codebert"
            self._provider_override = None
            self.dimension = CODEBERT_DIM
        elif provider_id and provider_id != "hash":
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

        if self.provider == "graphcodebert":
            vector = self._graphcodebert_embedding(identifier, text)
        elif self.provider == "codebert":
            vector = self._codebert_embedding(text)
        elif self.provider == "hash" or not self._registry or not self._provider_override:
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

    def _codebert_embedding(self, text: str) -> List[float]:
        """Generate CodeBERT embedding for code snippet."""
        global _CODEBERT_MODEL, _CODEBERT_TOKENIZER

        # Lazy load CodeBERT model
        if _CODEBERT_MODEL is None or _CODEBERT_TOKENIZER is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                print("Loading CodeBERT model (microsoft/codebert-base)...")
                _CODEBERT_TOKENIZER = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                _CODEBERT_MODEL = AutoModel.from_pretrained("microsoft/codebert-base")
                _CODEBERT_MODEL.eval()  # Set to evaluation mode
                print("CodeBERT model loaded successfully!")
            except Exception as e:
                print(f"Failed to load CodeBERT: {e}. Falling back to hash embedding.")
                return self._hash_embedding(text)

        try:
            import torch

            # Truncate text if too long (max 512 tokens for CodeBERT)
            text = text[:2000]  # Rough character limit

            # Tokenize and encode
            inputs = _CODEBERT_TOKENIZER(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            )

            # Generate embedding (mean pool over tokens)
            with torch.no_grad():
                outputs = _CODEBERT_MODEL(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                vector = embedding.tolist()

            return [float(v) for v in vector]

        except Exception as e:
            print(f"CodeBERT embedding failed: {e}. Falling back to hash.")
            return self._hash_embedding(text)

    def _graphcodebert_embedding(self, identifier: str, text: str) -> List[float]:
        """Generate GraphCodeBERT embedding for code snippet.

        Loads microsoft/graphcodebert-base model from bundled weights and generates
        embeddings on-the-fly for any code. This is required for the 91.8% GNN model.
        """
        global _GRAPHCODEBERT_MODEL, _GRAPHCODEBERT_TOKENIZER

        # Lazy load GraphCodeBERT model
        if _GRAPHCODEBERT_MODEL is None or _GRAPHCODEBERT_TOKENIZER is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                # Path to bundled model weights (local, no internet needed)
                bundled_model_path = Path(__file__).resolve().parent.parent.parent / "models" / "graphcodebert"

                if bundled_model_path.exists():
                    print(f"Loading GraphCodeBERT from bundled weights: {bundled_model_path}")
                    _GRAPHCODEBERT_TOKENIZER = AutoTokenizer.from_pretrained(str(bundled_model_path))
                    _GRAPHCODEBERT_MODEL = AutoModel.from_pretrained(str(bundled_model_path))
                else:
                    # Fallback: Download from HuggingFace (first-time only)
                    print("Bundled GraphCodeBERT not found. Downloading from microsoft/graphcodebert-base...")
                    print("(This is a one-time download. Bundle weights for offline use.)")
                    _GRAPHCODEBERT_TOKENIZER = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
                    _GRAPHCODEBERT_MODEL = AutoModel.from_pretrained("microsoft/graphcodebert-base")

                _GRAPHCODEBERT_MODEL.eval()  # Set to evaluation mode
                print("✅ GraphCodeBERT model loaded successfully!")
            except Exception as e:
                print(f"Failed to load GraphCodeBERT: {e}. Falling back to hash embedding.")
                return self._hash_embedding(text)

        try:
            import torch

            # Truncate text if too long (max 512 tokens for GraphCodeBERT)
            text = text[:2000]  # Rough character limit

            # Tokenize and encode
            inputs = _GRAPHCODEBERT_TOKENIZER(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            )

            # Generate embedding (mean pool over tokens)
            with torch.no_grad():
                outputs = _GRAPHCODEBERT_MODEL(**inputs)
                # Use [CLS] token embedding (first token) - same as CodeBERT
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                vector = embedding.tolist()

            return [float(v) for v in vector]

        except Exception as e:
            print(f"GraphCodeBERT embedding failed: {e}. Falling back to hash.")
            return self._hash_embedding(text)

    def _persist_cache(self) -> None:
        MAX_CACHE_ENTRIES = 10000
        if len(self._cache) > MAX_CACHE_ENTRIES:
            # Trim the cache, keeping the most recent entries.
            # This is a simple approach; a more complex LRU cache could be used in the future.
            keys_to_keep = list(self._cache.keys())[-MAX_CACHE_ENTRIES:]
            self._cache = {key: self._cache[key] for key in keys_to_keep}

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
