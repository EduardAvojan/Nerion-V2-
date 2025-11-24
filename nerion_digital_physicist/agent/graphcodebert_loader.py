"""Loader for pre-computed GraphCodeBERT embeddings.

This module provides utilities to load and access GraphCodeBERT embeddings
generated on GPU for the entire curriculum database.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_EMBEDDINGS_CACHE: Optional[Dict[str, torch.Tensor]] = None
_LESSON_ID_MAP: Optional[Dict[int, int]] = None
_LESSON_NAME_MAP: Optional[Dict[str, int]] = None


def load_graphcodebert_embeddings(embeddings_path: Optional[Path] = None) -> Tuple[Dict[str, torch.Tensor], Dict[int, int], Dict[str, int]]:
    """Load GraphCodeBERT embeddings from disk.

    Returns:
        Tuple of (embeddings_dict, lesson_id_map, lesson_name_map)
        - embeddings_dict: {'before_embeddings': Tensor, 'after_embeddings': Tensor}
        - lesson_id_map: {lesson_id -> index}
        - lesson_name_map: {lesson_name -> index}
    """
    global _EMBEDDINGS_CACHE, _LESSON_ID_MAP, _LESSON_NAME_MAP

    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE, _LESSON_ID_MAP, _LESSON_NAME_MAP  # type: ignore

    if embeddings_path is None:
        embeddings_path = Path(__file__).resolve().parent.parent.parent / "graphcodebert_embeddings.pt"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"GraphCodeBERT embeddings not found at {embeddings_path}")

    print(f"Loading GraphCodeBERT embeddings from {embeddings_path}...")
    data = torch.load(embeddings_path)

    _EMBEDDINGS_CACHE = {
        'before_embeddings': data['before_embeddings'],
        'after_embeddings': data['after_embeddings'],
        'dimension': data['dimension'],
        'model': data['model'],
    }

    # Build lookup maps
    _LESSON_ID_MAP = {}
    _LESSON_NAME_MAP = {}
    for idx, (lesson_id, lesson_name) in enumerate(zip(data['lesson_ids'], data['lesson_names'])):
        _LESSON_ID_MAP[lesson_id] = idx
        _LESSON_NAME_MAP[lesson_name] = idx

    print(f"✅ Loaded {len(data['lesson_ids'])} GraphCodeBERT embeddings (dimension={data['dimension']})")

    return _EMBEDDINGS_CACHE, _LESSON_ID_MAP, _LESSON_NAME_MAP


def get_lesson_embedding(lesson_id: Optional[int] = None, lesson_name: Optional[str] = None, sample_type: str = "before") -> Optional[List[float]]:
    """Get embedding for a specific lesson.

    Args:
        lesson_id: Database lesson ID
        lesson_name: Lesson name
        sample_type: 'before' or 'after'

    Returns:
        768-dimensional embedding vector or None if not found
    """
    embeddings, id_map, name_map = load_graphcodebert_embeddings()

    # Look up by ID or name
    idx = None
    if lesson_id is not None and lesson_id in id_map:
        idx = id_map[lesson_id]
    elif lesson_name is not None and lesson_name in name_map:
        idx = name_map[lesson_name]

    if idx is None:
        return None

    # Get embedding
    if sample_type == "before":
        embedding_tensor = embeddings['before_embeddings'][idx]
    elif sample_type == "after":
        embedding_tensor = embeddings['after_embeddings'][idx]
    else:
        raise ValueError(f"Invalid sample_type: {sample_type}. Must be 'before' or 'after'.")

    return embedding_tensor.tolist()


def get_embedding_for_code(code: str, lesson_context: Optional[str] = None) -> List[float]:
    """Get embedding for arbitrary code snippet.

    If lesson_context matches a known lesson, returns the GraphCodeBERT embedding.
    Otherwise, returns a deterministic hash-based embedding for consistency.

    Args:
        code: Source code
        lesson_context: Optional lesson name for lookup

    Returns:
        768-dimensional embedding vector
    """
    if lesson_context:
        # Try before/after lookup
        for sample_type in ["before", "after"]:
            emb = get_lesson_embedding(lesson_name=lesson_context, sample_type=sample_type)
            if emb is not None:
                return emb

    # Fallback: hash-based embedding (same as hash embedder)
    return _hash_embedding(code, dimension=768)


def _hash_embedding(text: str, dimension: int = 768) -> List[float]:
    """Generate deterministic hash-based embedding (fallback)."""
    seed = text.encode("utf-8")
    vector: List[float] = []
    for idx in range(dimension):
        hasher = hashlib.sha256()
        hasher.update(seed)
        hasher.update(idx.to_bytes(4, "little", signed=False))
        digest = hasher.digest()
        integer = int.from_bytes(digest[:8], "big", signed=False)
        value = (integer / (2**64 - 1)) * 2.0 - 1.0
        vector.append(float(value))
    return vector
