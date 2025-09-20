from __future__ import annotations

import hashlib
import math
import os
import re
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # numpy is available in the project dependencies
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def cosine(a: Optional["np.ndarray"], b: Optional["np.ndarray"]) -> float:
    if np is None:
        return 0.0
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BM25Lite:
    def __init__(self, docs: Sequence[Dict[str, Any]]) -> None:
        self.N = max(1, len(docs))
        self.df: Dict[str, int] = {}
        self.doc_tokens: List[set[str]] = []
        for doc in docs:
            tokens = set(tokenize(doc.get("fact", "")))
            self.doc_tokens.append(tokens)
            for token in tokens:
                self.df[token] = self.df.get(token, 0) + 1

    def score(self, idx: int, query_tokens: Iterable[str]) -> float:
        score = 0.0
        doc_tokens = self.doc_tokens[idx] if idx < len(self.doc_tokens) else set()
        for token in query_tokens:
            if token not in doc_tokens:
                continue
            df = self.df.get(token, 0)
            if df == 0:
                continue
            score += math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        return score


class DefaultEmbedder:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._model = None
        provider = (os.getenv("NERION_MEMORY_EMBEDDER") or "").strip()
        model = None
        if provider:
            path = provider
            if os.path.isdir(provider):
                path = provider
            env_allow = os.getenv("NERION_MEMORY_ALLOW_MODEL", "0").lower() in {"1", "true", "yes", "on"}
            if os.path.isdir(path) or env_allow:
                with suppress(Exception):
                    from sentence_transformers import SentenceTransformer  # type: ignore

                    model = SentenceTransformer(path)
        self._model = model

    def encode(self, texts: Sequence[str]):  # -> np.ndarray
        if np is None:
            return self._fallback(texts)
        if self._model is not None:
            with suppress(Exception):
                vectors = self._model.encode(list(texts), convert_to_numpy=True)  # type: ignore[attr-defined]
                if vectors is not None:
                    arr = np.asarray(vectors, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    return arr
        return self._fallback(texts)

    def _fallback(self, texts: Sequence[str]):
        if np is None:
            return [[0.0] for _ in texts]
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.dim
                arr[i, bucket] += 1.0
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms


class HybridRetriever:
    def __init__(self, docs: Sequence[Dict[str, Any]], embedder: Optional[DefaultEmbedder] = None) -> None:
        self.embedder = embedder or DefaultEmbedder()
        self._docs: List[Dict[str, Any]] = [doc for doc in docs if not doc.get("deleted")]
        self._bm25 = BM25Lite(self._docs)
        self._tokens = [tokenize(doc.get("fact", "")) for doc in self._docs]
        self._tags = [set(doc.get("tags") or []) for doc in self._docs]
        self._scopes = [doc.get("scope", "short") for doc in self._docs]
        self._embeddings = self._build_embeddings(self._docs)

    def _build_embeddings(self, docs: Sequence[Dict[str, Any]]):
        if np is None or not docs:
            return None
        texts = [doc.get("fact", "") for doc in docs]
        return self.embedder.encode(texts)

    def topk(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._docs or not query.strip():
            return []
        query_tokens = tokenize(query)
        if np is not None:
            query_vec = self.embedder.encode([query])[0]
        else:
            query_vec = None
        scored: List[tuple[float, Dict[str, Any]]] = []
        for idx, doc in enumerate(self._docs):
            bm_score = self._bm25.score(idx, query_tokens)
            emb_score = 0.0
            if np is not None and self._embeddings is not None:
                emb_score = cosine(self._embeddings[idx], query_vec)
            tag_bonus = 0.1 if self._tags[idx] & set(query_tokens) else 0.0
            scope_bonus = 0.2 if self._scopes[idx] == "long" else 0.0
            combined = 0.5 * bm_score + 0.4 * emb_score + tag_bonus + scope_bonus
            scored.append((combined, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [doc for score, doc in scored[: max(0, k)] if score > 0.0]
        return top
