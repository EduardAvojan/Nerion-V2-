"""Shared intent slot parser used by runner and engine.

Deterministic, offline rules to extract a coarse intent and slots
from a free-form utterance. Centralized to avoid drift across modules.
"""
from __future__ import annotations
import re
from typing import Optional, Dict

_TASK_WORDS = {
    "best_pick": ("best", "top", "recommend"),
    "fact_lookup": ("current", "latest", "now", "today", "recent"),
    "compare": ("compare", "vs"),
    "summarize": ("summarize", "read", "digest"),
}

_TIMEFRAMES = {
    "this month": ("this month", "for the month", "month"),
    "today": ("today", "right now"),
    "this week": ("this week",),
    "this quarter": ("this quarter", "quarter"),
}

_LOC_RX = re.compile(
    r"\b(?:in|near|around)\s+([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+)*)\b",
    re.IGNORECASE,
)
_ITEM_RX = re.compile(r"\b(best|top|recommend|current|latest|compare|vs|summarize)\s+(?:the\s+)?([a-zA-Z0-9\- ]+)\b")
_DOMAIN_RX = re.compile(r"(https?://[^\s]+|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})")


def parse_task_slots(text: str) -> Optional[Dict]:
    """Parse a free-form utterance into {task, item, location, timeframe, sources, confidence}."""
    if not text or not text.strip():
        return None
    t = text.strip()
    low = t.lower()
    # Task detection by keywords
    task = None
    for tk, kws in _TASK_WORDS.items():
        if any(k in low for k in kws):
            task = tk
            break
    if not task:
        return None
    # Sources (domains) mentioned in utterance
    sources = [m[0] if isinstance(m, tuple) else m for m in _DOMAIN_RX.findall(t)]
    # Timeframe
    timeframe = None
    for tf, kws in _TIMEFRAMES.items():
        if any(k in low for k in kws):
            timeframe = tf
            break
    # Location
    loc_m = _LOC_RX.search(t)
    location = loc_m.group(1) if loc_m else None
    # Item (head noun following task cue)
    item = None
    m = _ITEM_RX.search(t)
    if m:
        item = (m.group(2) or "").strip(" .,")
    # Confidence: simple sum of matched slots
    score = 0.0
    score += 0.4  # task hit
    if item:
        score += 0.25
    if timeframe:
        score += 0.2
    if location:
        score += 0.15
    if sources:
        score += 0.1
    conf = min(1.0, score)
    return {
        "intent": task,
        "item": item,
        "location": location,
        "timeframe": timeframe,
        "sources": sources,
        "default_temperature": 0.2 if task in ("best_pick", "fact_lookup", "summarize") else 0.7,
        "confidence": conf,
    }

