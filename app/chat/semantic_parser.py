from __future__ import annotations

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Offline-first semantic intent classifier (data-driven, no network usage)
# --------------------------------------------------------------------------------------
# Usage pattern:
#   1) Call configure_intents(examples=...) once at startup with a dict mapping
#      intent -> list[str] examples. (Load this from data, not code.)
#   2) Optionally call configure_model(local_model_dir=...) to load a local
#      sentence-transformers model (no downloads). If not provided or fails,
#      we fall back to fuzzy matching.
#   3) Call parse_intent_by_similarity(text, threshold) to get best intent or None.
# --------------------------------------------------------------------------------------

# In-memory state
_examples: Dict[str, List[str]] = {}
_model = None  # sentence-transformers model (optional)
_intent_embs: Dict[str, Any] = {}  # lazily built tensors; typed as Any to avoid hard dependency


def configure_intents(*, examples: Dict[str, List[str]]) -> None:
    """Set/replace intent examples from data (YAML/JSON). No strings hardcoded here.
    Clears cached embeddings so they rebuild on next use.
    """
    global _examples, _intent_embs
    _examples = {k: list(v or []) for k, v in (examples or {}).items()}
    _intent_embs = {}


def configure_model(*, local_model_dir: Optional[str] = None) -> bool:
    """Attempt to load a local sentence-transformers model from a directory.
    Returns True if loaded successfully; False if not available (fallback will be used).
    NEVER downloads; strictly offline.
    """
    global _model, _intent_embs
    _model = None
    _intent_embs = {}
    if not local_model_dir:
        logger.info("semantic_parser: no local_model_dir provided; using fuzzy fallback")
        return False
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer(local_model_dir)
        logger.info("semantic_parser: loaded local ST model from %s", local_model_dir)
        return True
    except Exception as e:  # pragma: no cover
        logger.warning("semantic_parser: could not load model from %s: %s", local_model_dir, e)
        _model = None
        return False


def _ensure_intent_embeddings() -> bool:
    """Build embeddings for current examples if model is present. Returns True if ready."""
    global _intent_embs
    if _model is None:
        return False
    if _intent_embs:
        return True
    try:
        _intent_embs = {intent: _model.encode(exs, convert_to_tensor=True) for intent, exs in _examples.items()}
        return True
    except Exception as e:  # pragma: no cover
        logger.warning("semantic_parser: embedding build failed: %s", e)
        _intent_embs = {}
        return False


def parse_intent_by_similarity(text: str, *, threshold: float = 0.55) -> Optional[str]:
    """Return best-matching intent using local embeddings if available; otherwise fuzzy.
    - threshold applies to cosine similarity if embeddings are available, else to scaled
      fuzzy score (0..1) derived from RapidFuzz ratio (0..100).
    """
    if not text or not _examples:
        return None

    # Path A: sentence-transformers (strictly offline, local)
    if _model is not None and _ensure_intent_embeddings():
        try:
            from sentence_transformers import util  # type: ignore
            emb = _model.encode(text, convert_to_tensor=True)
            best_intent: Optional[str] = None
            best_score = -1.0
            for intent, ex_mat in _intent_embs.items():
                scores = util.cos_sim(emb, ex_mat)[0]
                score = float(scores.max().item())
                if score > best_score:
                    best_score = score
                    best_intent = intent
            if best_intent is not None and best_score >= threshold:
                logger.debug("semantic_parser: ST match %s (%.2f)", best_intent, best_score)
                return best_intent
        except Exception as e:  # pragma: no cover
            logger.warning("semantic_parser: ST match failed: %s", e)
            # fall through to fuzzy

    # Path B: Fuzzy over examples (no ML, no downloads)
    # Prefer RapidFuzz if available; otherwise use stdlib/dumb fallback.
    _have_rf = True
    try:
        from rapidfuzz import fuzz  # type: ignore
    except Exception:
        _have_rf = False
        fuzz = None  # type: ignore

    text_norm = str(text).strip().lower()
    best_intent: Optional[str] = None
    best_score = -1.0
    if _have_rf:
        for intent, exs in _examples.items():
            for ex in exs:
                ex_norm = str(ex).strip().lower()
                try:
                    s1 = float(fuzz.token_set_ratio(text_norm, ex_norm)) / 100.0
                    s2 = float(fuzz.token_sort_ratio(text_norm, ex_norm)) / 100.0
                    s3 = float(fuzz.partial_ratio(text_norm, ex_norm)) / 100.0
                    try:
                        s4 = float(fuzz.WRatio(text_norm, ex_norm)) / 100.0
                    except Exception:
                        s4 = 0.0
                    s = max(s1, s2, s3, s4)
                except Exception:
                    s = 0.0
                if s > best_score:
                    best_score = s
                    best_intent = intent
    else:
        # Stdlib-only fallback: difflib ratio + simple token overlap
        import re
        import difflib
        def _tokens(s: str) -> set[str]:
            return set(re.findall(r"\w+", s))
        tset = _tokens(text_norm)
        for intent, exs in _examples.items():
            for ex in exs:
                ex_norm = str(ex).strip().lower()
                try:
                    s_dif = difflib.SequenceMatcher(None, text_norm, ex_norm).ratio()
                except Exception:
                    s_dif = 0.0
                eset = _tokens(ex_norm)
                inter = len(tset & eset)
                denom = len(eset) or 1
                # coverage = fraction of example tokens covered by text (biased to exact phrasing examples)
                s_cov = inter / denom
                s = max(s_dif, s_cov)
                if s > best_score:
                    best_score = s
                    best_intent = intent
    if best_intent is not None and best_score >= threshold:
        logger.debug("semantic_parser: fuzzy match %s (%.2f)", best_intent, best_score)
        return best_intent
    return None