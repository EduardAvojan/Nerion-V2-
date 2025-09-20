"""
This module handles consensus aggregation of multiple evidence sources,
including onsite and external reviews.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import time
import re

_ACRONYMS = {"ai", "pc", "vr", "xr", "ml", "nlp", "gpu", "cpu", "ssd", "oled"}
_STOP_BEGINS = (
    "what's", "whats", "make every", "learn more", "shop now", "compare", "explore",
)
_GENERIC_TAILS = {"index", "home", "products", "category"}


def _is_productlike(name: str) -> bool:
    nm = (name or "").strip()
    # contains a digit or a model-ish token like G1/14/etc.
    return bool(re.search(r"\d", nm)) or bool(re.search(r"\b[a-zA-Z]\d|\d[a-zA-Z]\b", nm))


def _is_categorylike(name: str) -> bool:
    nm = (name or "").strip().lower()
    return any(k in nm for k in ("pcs", "products", "solutions", "next gen"))


def _productness_bonus(name: str) -> float:
    """Small, transparent nudge toward product models over categories."""
    bonus = 0.0
    if _is_productlike(name):
        bonus += 0.05
    if _is_categorylike(name):
        bonus -= 0.03
    return bonus


def _fix_acronyms(title: str) -> str:
    words = title.title().split()
    fixed = []
    for w in words:
        lw = w.lower()
        if lw in _ACRONYMS:
            fixed.append(lw.upper())
        elif lw.endswith("s") and lw[:-1] in _ACRONYMS:
            fixed.append(lw[:-1].upper() + "s")
        else:
            fixed.append(w)
    return " ".join(fixed)


def _clean_title(s: str, max_words: int = 6, max_chars: int = 60) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return ""
    # remove leading boilerplate phrases
    low = s.lower()
    for pref in _STOP_BEGINS:
        if low.startswith(pref):
            s = s[len(pref):].strip()
            break
    # limit words/chars and fix acronyms
    words = s.split()
    s = " ".join(words[:max_words])
    s = s[:max_chars].rstrip()
    return _fix_acronyms(s)


def _name_from_summary_or_url(ev: Dict[str, Any]) -> str:
    """Derive a compact, readable candidate name from URL or summary.

    Preference order: explicit name > URL tail > summary head.
    Applies acronym casing and trims generic/long phrases.
    """
    # Prefer explicit
    if ev.get("name"):
        name = _clean_title(str(ev["name"]))
        if name:
            return name
    # Try URL tail
    url = (ev.get("url") or "").strip()
    if url:
        tail = url.rstrip("/").split("/")[-1]
        tail = re.sub(r"\.(html?|php|aspx)$", "", tail, flags=re.IGNORECASE)
        tail = re.sub(r"[\-_]+", " ", tail)
        tail = tail.strip()
        if tail and tail.lower() not in _GENERIC_TAILS:
            name = _clean_title(tail)
            if name:
                return name
    # Fall back to first sentence of summary
    summ = (ev.get("summary") or "").strip()
    if summ:
        head = re.split(r"[.!?]", summ)[0].strip()
        name = _clean_title(head)
        if name:
            return name
    return "candidate"


def _epoch_from_yyyy_mm_dd(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        tm = time.strptime(s, "%Y-%m-%d")
        return time.mktime(tm)
    except Exception:
        return None


def aggregate_consensus(evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate top evidences into a consensus structure.

    Inputs (per evidence item):
      - summary: str (short text snippet)
      - confidence: float (0..1)
      - source: optional 'onsite' | 'external'
      - url: optional str
      - date_hint: optional 'YYYY-MM-DD'
      - latest_epoch: optional float
      - name: optional str

    Output shape:
      {
        "summary": <joined unique summaries>,
        "confidence": <avg confidence>,
        "consensus": <same as summary>,
        "source_count": N,
        "winner": { name, score, source, url, date_hint },
        "candidates": [ { name, score, source, url, date_hint }, ... ],
        "method": "generic_v1"
      }
    """
    if not evidences:
        return {
            "summary": "",
            "confidence": 0.0,
            "consensus": "",
            "source_count": 0,
            "winner": {},
            "candidates": [],
            "method": "generic_v1",
        }

    # Build candidate list with transparent scores
    cands: List[Dict[str, Any]] = []
    total_conf = 0.0
    seen_summ: List[str] = []
    for ev in evidences:
        conf = float(ev.get("confidence") or 0.0)
        total_conf += conf
        src = (ev.get("source") or "").lower() or "onsite"
        url = ev.get("url")
        date_hint = ev.get("date_hint")
        lepoch = ev.get("latest_epoch") or _epoch_from_yyyy_mm_dd(date_hint)
        name = _name_from_summary_or_url(ev)
        # base score on provided confidence
        score = conf
        # external support bonus (very small)
        if src == "external":
            score += 0.02
        # recency bonus (if newer than ~6 months vs zero), tiny nudge
        if isinstance(lepoch, (int, float)):
            age_days = max(0.0, (time.time() - float(lepoch)) / 86400.0)
            if age_days < 180:
                score += 0.02
        # product vs category nudge (very small)
        score += _productness_bonus(name)
        cands.append({
            "name": name,
            "score": round(score, 6),
            "source": src,
            "url": url,
            "date_hint": date_hint,
        })

        summ = (ev.get("summary") or "").strip()
        if summ and summ not in seen_summ:
            seen_summ.append(summ)

    # Sort by score desc
    cands.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Tie-break: if two are close (<= 0.01), prefer external then recency (date_hint present)
    if len(cands) >= 2:
        top, nxt = cands[0], cands[1]
        if abs(top["score"] - nxt["score"]) <= 0.01:
            # prefer external
            if nxt.get("source") == "external" and top.get("source") != "external":
                cands[0], cands[1] = nxt, top
            else:
                swapped = False
                # prefer one with date_hint
                if nxt.get("date_hint") and not top.get("date_hint"):
                    cands[0], cands[1] = nxt, top
                    swapped = True
                # prefer productlike over categorylike if still close
                if not swapped:
                    if _is_productlike(nxt.get("name", "")) and _is_categorylike(top.get("name", "")):
                        cands[0], cands[1] = nxt, top

    # Winner and summary
    winner = cands[0] if cands else {}
    avg_confidence = total_conf / max(len(evidences), 1)
    consensus_text = " / ".join(seen_summ)

    return {
        "summary": consensus_text,
        "confidence": avg_confidence,
        "consensus": consensus_text,
        "source_count": len(evidences),
        "winner": winner,
        "candidates": cands,
        "method": "generic_v1",
    }
