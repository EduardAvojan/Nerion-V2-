"""Conversation context utilities for Nerion.
Holds normalization, token similarity, topic relation, and ambiguity checks.
"""
from __future__ import annotations
import re
from typing import Set

__all__ = [
    "_normalize",
    "_tokenize",
    "_jaccard",
    "_recent_context_text",
    "_sim_score",
    "_auto_title",
    "_relation_to_context",
    "AMBIGUOUS_TERMS",
    "DISAMBIG_HINTS",
    "_needs_disambiguation",
    "CONVO_WARM_TTL_S",
    "SIM_ON_THRESHOLD",
    "SIM_OFF_THRESHOLD",
]

# --- Similarity thresholds and warmth window (used by relation logic) ---------
CONVO_WARM_TTL_S = 20 * 60  # 20 minutes
SIM_ON_THRESHOLD = 0.28
SIM_OFF_THRESHOLD = 0.15

# --- Basic normalization and token similarity ---------------------------------

def _normalize(text: str) -> str:
    return ' '.join(re.sub(r'[^a-z0-9\s]', ' ', (text or '').lower()).split())


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return set(_normalize(text).split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


# --- Context plumbing ---------------------------------------------------------

def _recent_context_text(state, n_turns: int = 6) -> str:
    try:
        active = state.active
        if not active:
            return ''
        topic = getattr(active, 'topic', '') or ''
        hist = list(getattr(active, 'chat_history', []) or [])
        tail = hist[-n_turns:]
        parts = [topic] + [f"{t.get('role','')}: {t.get('content','')}" for t in tail]
        return '\n'.join(parts)
    except Exception:
        return ''


def _sim_score(user_text: str, state) -> float:
    ctx = _recent_context_text(state)
    return _jaccard(_tokenize(user_text), _tokenize(ctx))


def _auto_title(text: str, max_len: int = 60) -> str:
    t = (text or '').strip()
    if len(t) <= max_len:
        return t
    return t[:max_len-1] + '…'


def _relation_to_context(user_text: str, state) -> str:
    """Return 'on', 'off', or 'ambiguous' based on similarity and warmth.
    (Warmth based on last_touched is handled elsewhere; we use similarity only.)
    """
    s = _sim_score(user_text, state)
    if s >= SIM_ON_THRESHOLD:
        return 'on'
    if s <= SIM_OFF_THRESHOLD:
        return 'off'
    return 'ambiguous'


# --- Ambiguity handling -------------------------------------------------------
AMBIGUOUS_TERMS = {
    'apple', 'java', 'mercury', 'python', 'rust', 'go', 'cuda', 'torch', 'jaguar', 'merlin'
}
DISAMBIG_HINTS = {
    'company', 'fruit', 'language', 'metal', 'planet', 'programming', 'framework', 'gpu', 'animal', 'car'
}


def _needs_disambiguation(text: str) -> bool:
    low = (text or '').lower()
    if not low:
        return False
    if any(term in low.split() for term in AMBIGUOUS_TERMS):
        if not any(h in low for h in DISAMBIG_HINTS):
            return True
    return False