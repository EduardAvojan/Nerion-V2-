from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable
from urllib.parse import urlparse
import re
import time

MIN_TEXT_CHARS = 2500  # drop very thin pages (avoid link hubs, nav pages)

_MONTHS = {
    'jan': 1, 'january': 1,
    'feb': 2, 'february': 2,
    'mar': 3, 'march': 3,
    'apr': 4, 'april': 4,
    'may': 5,
    'jun': 6, 'june': 6,
    'jul': 7, 'july': 7,
    'aug': 8, 'august': 8,
    'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}

_DEFRESH = '60d'


def _friendly_missing(name: str, extra: str) -> RuntimeError:
    return RuntimeError(
        f"External augmentation requires '{name}'. Install extras: pip install -e '.[{extra}]'"
    )


def _is_http(url: str) -> bool:
    return url.lower().startswith("http://") or url.lower().startswith("https://")


def _normalize_allow(allow: Optional[Iterable[str]]) -> List[str]:
    out: List[str] = []
    if not allow:
        return out
    for item in allow:
        if not item:
            continue
        s = str(item).strip()
        if _is_http(s):
            out.append(s)
        else:
            out.append(f"https://{s}")
    return out


def _same_host(a: str, b: str) -> bool:
    try:
        pa = urlparse(a).netloc.split(":")[0].lower()
        pb = urlparse(b).netloc.split(":")[0].lower()
        return bool(pa and pb) and pa == pb
    except Exception:
        return False


def _html_to_text(html: str) -> str:
    """Super-light HTML -> text using regex + space compaction."""
    if not html:
        return ""
    # Remove scripts/styles
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    # Strip tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html).strip()
    return html


def _parse_window(spec: Optional[str]) -> Optional[float]:
    """Return a cutoff epoch (now - delta) for spec like '60d', '6m', '1y'. None if not provided."""
    if not spec:
        return None
    spec = spec.strip().lower()
    m = re.match(r"^(\d+)([dmy])$", spec)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    days = n
    if unit == 'm':
        days = n * 30
    elif unit == 'y':
        days = n * 365
    cutoff = time.time() - days * 86400
    return cutoff


def _latest_epoch(text: str) -> Optional[float]:
    """Extract the most recent date hint (very light regex)."""
    if not text:
        return None
    candidates: list[float] = []
    # ISO yyyy-mm-dd
    for y, mo, d in re.findall(r"\b(20\d{2}|19\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", text):
        try:
            tm = time.strptime(f"{y}-{mo}-{d}", "%Y-%m-%d")
            candidates.append(time.mktime(tm))
        except Exception:
            pass
    # ISO yyyy-mm
    for y, mo in re.findall(r"\b(20\d{2}|19\d{2})-(0[1-9]|1[0-2])\b", text):
        try:
            tm = time.strptime(f"{y}-{mo}-01", "%Y-%m-%d")
            candidates.append(time.mktime(tm))
        except Exception:
            pass
    # Month name yyyy
    for mon, y in re.findall(r"\b([A-Za-z]{3,9})\s+(20\d{2}|19\d{2})\b", text):
        try:
            mnum = _MONTHS.get(mon.lower())
            if mnum:
                tm = time.strptime(f"{y}-{mnum:02d}-01", "%Y-%m-%d")
                candidates.append(time.mktime(tm))
        except Exception:
            pass
    if not candidates:
        return None
    return max(candidates)


def _retry_get(url: str, *, timeout: int, retries: int = 2, backoff: float = 0.5) -> Optional[str]:
    """Best-effort GET with small retries and exponential backoff. Returns text or None."""
    try:
        import requests
    except Exception:
        return None
    headers = {"User-Agent": "nerion-augment/1.0"}
    _last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            _last_exc = e
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                return None
    return None


def gather_external(
    query: str,
    *,
    root_host: str,
    allow: Optional[Iterable[str]] = None,
    block: Optional[Iterable[str]] = None,
    max_pages: int = 6,
    timeout: int = 10,
    render: bool = False,
    fresh_within: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Gather a small set of external evidence pages (generic, domain-agnostic).

    Parameters
    ----------
    query : str
        User intent text to guide selection (currently used for mild scoring only).
    root_host : str
        Host of the primary site; external evidence should not be the same host.
    allow : Iterable[str] | None
        Optional list of domains or full URLs to pull. Bare domains become https://<domain>.
    block : Iterable[str] | None
        Optional list of substrings to exclude.
    max_pages : int
        Upper bound pages to fetch.
    timeout : int
        Network timeout per request.
    render : bool
        If True, use web_render (Playwright) to fetch; else try requests.
    fresh_within : Optional[str]
        If provided, require detected dates in content within this window (e.g. '60d').

    Returns
    -------
    List[Dict[str, Any]]: evidence items with keys {url, source, text}.
    """
    # Guarded imports
    if render:
        try:
            from selfcoder.analysis import web_render  # noqa: F401
        except Exception as e:
            raise _friendly_missing("playwright/web_render", "docs-web") from e
    else:
        try:
            import requests  # noqa: F401
        except Exception as e:
            raise _friendly_missing("requests", "docs-web") from e

    cutoff = _parse_window(fresh_within)

    # Normalize allow list -> candidate URLs
    candidates: List[str] = _normalize_allow(allow)
    # Deduplicate & filter
    seen: set[str] = set()
    items: List[Dict[str, Any]] = []

    # Helper to decide if blocked
    def is_blocked(u: str) -> bool:
        if not block:
            return False
        ul = u.lower()
        return any(b.strip().lower() in ul for b in block if b)

    # Fetch function
    def fetch(url: str) -> Optional[str]:
        try:
            if render:
                from selfcoder.analysis import web_render
                return web_render.render_url(url, timeout=timeout, render_timeout=max(5, timeout // 2))
            else:
                return _retry_get(url, timeout=timeout)
        except Exception:
            return None

    # If user provided full URLs, use them as-is; if only domains, fetch homepage first.
    fetched = 0
    for u in candidates:
        if fetched >= max_pages:
            break
        u_clean = u.strip()
        if not _is_http(u_clean):
            continue
        if is_blocked(u_clean):
            continue
        if _same_host(u_clean, root_host):
            # skip same host; external only
            continue
        if u_clean in seen:
            continue
        seen.add(u_clean)
        html = fetch(u_clean)
        if not html:
            continue
        text = _html_to_text(html)
        if not text:
            continue
        # Drop thin content even if recent
        if len(text) < MIN_TEXT_CHARS:
            continue
        # Freshness filter: require a detectable date within window if cutoff provided
        if cutoff is not None:
            latest = _latest_epoch(text)
            if latest is None or latest < cutoff:
                continue
        # Annotate evidence with date/fetch metadata
        latest = _latest_epoch(text)
        date_hint = time.strftime("%Y-%m-%d", time.gmtime(latest)) if latest else None
        items.append({
            "url": u_clean,
            "source": "external",
            "text": text,
            "latest_epoch": latest,
            "date_hint": date_hint,
            "fetched_at": time.time(),
        })
        fetched += 1

    return items

# Public alias for tests/consumers
html_to_text = _html_to_text

__all__ = [
    "MIN_TEXT_CHARS",
    "_parse_window",
    "_latest_epoch",
    "_html_to_text",
    "html_to_text",
    "gather_external",
]