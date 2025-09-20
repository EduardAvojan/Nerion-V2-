from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Optional, Tuple, Iterable

# Canonical domain labels we currently support
DOMAINS = (
    "finance",
    "real_estate",
    "healthcare",
    "tech_news",
    "world_news",
    "site_overview",
    "site_query",
    "general_topic",
)

_FINANCE = {
    "stock", "stocks", "ticker", "tickers", "market", "nasdaq", "nyse",
    "gainer", "loser", "portfolio", "earnings", "guidance",
}
_REAL_ESTATE = {
    "real estate", "housing", "house", "home", "homes", "mortgage",
    "rent", "rents", "rental", "zip", "neighborhood", "county",
}
_HEALTHCARE = {
    "drug", "drugs", "treatment", "trial", "phase", "fda", "ema",
    "alzheimer", "alzheimer's", "oncology", "diabetes", "biologic",
    "label", "approval", "clinical", "endpoint", "dose",
}
_NEWS = {
    "breaking", "headline", "headlines", "world", "today", "latest",
    "just in", "developing",
}
_TECH = {
    "tech", "technology", "software", "hardware", "ai", "semiconductor",
}

_URL_RE = re.compile(r"https?://", re.I)


def _has_url(text: str) -> bool:
    return bool(_URL_RE.search(text))


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _url_domain(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    try:
        return urlparse(u).netloc.lower() or None
    except Exception:
        return None


def _contains_any(text: str, keywords: set[str]) -> bool:
    t = _clean(text)
    return any(k in t for k in keywords)


# Helper for arbitrary token iterables (for hints)
def _contains_any_tokens(text: str, tokens: Iterable[str]) -> bool:
    t = _clean(text)
    for tok in tokens or ():  # defensive
        if not tok:
            continue
        if str(tok).lower() in t:
            return True
    return False


def classify_query(query: str = "", *, url: Optional[str] = None, hints: Optional[dict[str, Iterable[str]]] = None) -> Tuple[str, float]:
    """
    Return a (domain_label, confidence) for a free-form query and optional URL.

    Heuristic, fast, and dependency-free. Confidence is a rough 0..1 score.
    """
    q = _clean(query)
    has_url = bool(url) or _has_url(q)

    # Optional hints (per-profile classification boosts); prefer if provided
    if hints:
        matched: list[tuple[str, int]] = []
        for label, toks in hints.items():
            try:
                cnt = sum(1 for tok in toks if tok and str(tok).lower() in q)
            except Exception:
                cnt = 0
            if cnt > 0:
                matched.append((label, cnt))
        if matched:
            matched.sort(key=lambda x: x[1], reverse=True)
            top_label, top_count = matched[0]
            # Confidence scales mildly with token hits
            conf = 0.88 if top_count >= 2 else 0.82
            return (top_label, conf)

    # URL-only instruction => site overview
    if has_url and not q:
        return ("site_overview", 0.9)

    # If both URL and task words, likely a site_query
    if has_url and any(w in q for w in ("best", "find", "compare", "models", "specs", "summary", "post", "item")):
        return ("site_query", 0.85)

    # Count domain token matches and choose the dominant domain (tie-breaker favors finance > real_estate > healthcare)
    counts = {
        "finance": sum(1 for tok in _FINANCE if tok in q),
        "real_estate": sum(1 for tok in _REAL_ESTATE if tok in q),
        "healthcare": sum(1 for tok in _HEALTHCARE if tok in q),
    }
    max_count = max(counts.values()) if counts else 0
    if max_count > 0:
        # tie-break order
        order = ["finance", "real_estate", "healthcare"]
        winners = [k for k, v in counts.items() if v == max_count]
        for label in order:
            if label in winners:
                # mild confidence scaling with matches
                conf = 0.78 + min(0.1, 0.02 * (max_count - 1))
                return (label, conf)

    # Domain-specific signals (prioritize real estate over finance when both match)
    if _contains_any(q, _REAL_ESTATE):
        return ("real_estate", 0.8)
    if _contains_any(q, _FINANCE):
        return ("finance", 0.8)
    if _contains_any(q, _HEALTHCARE):
        return ("healthcare", 0.8)

    # News split: tech vs world
    if _contains_any(q, _NEWS) and _contains_any(q, _TECH):
        return ("tech_news", 0.75)
    if _contains_any(q, _NEWS):
        return ("world_news", 0.7)

    # If it’s explicitly about a known tech brand/site
    if has_url:
        host = _url_domain(url) or ""
        if any(b in host for b in ("apple.com", "hp.com", "microsoft.com", "google.com")):
            return ("site_overview", 0.7 if not q else 0.8)

    # Fallbacks based on instruction style
    if any(w in q for w in ("summarize", "overview", "what is", "about")):
        return ("general_topic", 0.6)

    # Default catch-all
    return ("general_topic", 0.5)


__all__ = ["classify_query", "DOMAINS"]