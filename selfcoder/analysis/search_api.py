from __future__ import annotations
import os
import logging
import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from ops.security.net_gate import NetworkGate

try:
    import requests  # lightweight, already in many environments
except Exception:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

# Providers: bing (Bing Web Search), serpapi (Google via SerpAPI), gcs (Google Custom Search JSON),
# duck (DuckDuckGo Instant Answer API as a very light fallback — not scraping SERP pages)
# NOTE: We intentionally DO NOT fetch or render Google HTML pages.

_GOOGLE_HOSTS = {
    "google.com", "www.google.com", "news.google.com", "maps.google.com",
    "consent.google.com", "support.google.com"
}


def normalize_search_env() -> tuple[str, bool]:
    """Normalize legacy env to current keys and ensure a default provider.

    - Maps SERPAPI_API_KEY -> NERION_SEARCH_API_KEY when the latter is unset.
    - Sets NERION_SEARCH_PROVIDER to 'serpapi' if a key is present, else 'duck'.
    Returns (provider, has_api_key).
    """
    try:
        prov = (os.getenv("NERION_SEARCH_PROVIDER") or "").strip().lower()
        api_key = (os.getenv("NERION_SEARCH_API_KEY") or "").strip()
        serpapi_key = (os.getenv("SERPAPI_API_KEY") or "").strip()
        if (not api_key) and serpapi_key:
            os.environ["NERION_SEARCH_API_KEY"] = serpapi_key
            api_key = serpapi_key
            if not prov:
                os.environ["NERION_SEARCH_PROVIDER"] = "serpapi"
                prov = "serpapi"
        if not prov:
            os.environ["NERION_SEARCH_PROVIDER"] = "serpapi" if api_key else "duck"
            prov = os.environ["NERION_SEARCH_PROVIDER"].strip().lower()
        return prov, bool(api_key)
    except Exception:
        # best-effort only; leave env untouched
        return (os.getenv("NERION_SEARCH_PROVIDER", "") or "").strip().lower(), bool(os.getenv("NERION_SEARCH_API_KEY", "").strip())


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and v.strip()) else default


def _clean_url(u: str) -> Optional[str]:
    try:
        u = (u or "").strip()
        if not u or u.startswith("/"):
            return None
        p = urlparse(u)
        if not p.scheme or not p.netloc:
            return None
        if p.netloc in _GOOGLE_HOSTS:
            return None
        if p.scheme not in {"http", "https"}:
            return None
        return f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
    except Exception:
        return None


def _dedupe_keep_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        cu = _clean_url(u)
        if not cu or cu in seen:
            continue
        seen.add(cu)
        out.append(cu)
    return out


def _map_freshness(freshness: Optional[str]) -> Optional[str]:
    if not freshness:
        return None
    f = freshness.lower().strip()
    if f in {"day", "daily", "1d", "today"}:
        return "Day"
    if f in {"week", "weekly", "7d", "this_week"}:
        return "Week"
    if f in {"month", "monthly", "30d", "this_month"}:
        return "Month"
    return None


def _allowed(u: str, allow: Optional[List[str]]) -> bool:
    if not allow:
        return True
    try:
        host = urlparse(u).netloc
        return any(host.endswith(dom) for dom in allow)
    except Exception:
        return True


_ATOMIC_MAX_LEN = 120
_KEY_MAX_LEN = 40
_MAX_PAIRS = 12

def _is_htmlish(s: str) -> bool:
    return bool(re.search(r"<[^>]+>|</|data:|base64,", s))

def _looks_like_url(s: str) -> bool:
    return bool(re.match(r"https?://", s))

def _normalize_key(k: str) -> str:
    k = (k or "").strip().replace("_", " ").replace("-", " ")
    if not k:
        return "Field"
    # Prefer the last path-like token to avoid very long prefixes
    parts = [p for p in re.split(r"[./>]|\\|\s+", k) if p]
    cand = parts[-1] if parts else k
    # Keep letters, numbers, space, percent, and the degree symbol for common units
    cand = re.sub(r"[^A-Za-z0-9 %°]", " ", cand)
    cand = re.sub(r"\s+", " ", cand).strip().title()
    if len(cand) > _KEY_MAX_LEN:
        cand = cand[:_KEY_MAX_LEN].rstrip()
    return cand or "Field"

def _value_ok(v: str) -> bool:
    if not v or not isinstance(v, str):
        return False
    v = v.strip()
    if not v or len(v) > _ATOMIC_MAX_LEN:
        return False
    if _is_htmlish(v) or _looks_like_url(v):
        return False
    # Generic signals of atomic facts: digits, units/symbols, short categorical labels
    if re.search(r"\d", v):
        return True
    if re.search(r"[%°°]|\b(km|mi|mph|kph|h|m|s)\b", v, re.I):
        return True
    # Short categorical (1-3 words, letters only)
    if len(v) <= 30 and bool(re.match(r"^[A-Za-z][A-Za-z ]{0,29}$", v)):
        return True
    return False

# Sections that are provider metadata/telemetry and should be ignored during flattening
_META_SECTIONS = {
    "search_metadata",
    "search_parameters",
    "search_information",
    "serpapi_pagination",
    "pagination",
}

def _flatten_pairs(obj: Any, prefix: str = "") -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    try:
        if isinstance(obj, dict):
            for k, val in obj.items():
                # Skip provider meta/telemetry sections at the top level (topic-agnostic)
                if not prefix and str(k) in _META_SECTIONS:
                    continue
                key_path = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(val, (dict, list)):
                    pairs.extend(_flatten_pairs(val, key_path))
                else:
                    if isinstance(val, (int, float)):
                        pairs.append({"key": _normalize_key(key_path), "value": str(val)})
                    elif isinstance(val, str) and _value_ok(val):
                        pairs.append({"key": _normalize_key(key_path), "value": val.strip()})
        elif isinstance(obj, list):
            for idx, val in enumerate(obj):
                key_path = f"{prefix}.{idx}" if prefix else str(idx)
                if isinstance(val, (dict, list)):
                    pairs.extend(_flatten_pairs(val, key_path))
                else:
                    if isinstance(val, (int, float)):
                        pairs.append({"key": _normalize_key(key_path), "value": str(val)})
                    elif isinstance(val, str) and _value_ok(val):
                        pairs.append({"key": _normalize_key(key_path), "value": val.strip()})
    except Exception:
        return []
    # de-dup while keeping order by (key,value)
    seen = set()
    out: List[Dict[str, str]] = []
    for p in pairs:
        t = (p.get("key", ""), p.get("value", ""))
        if t in seen:
            continue
        seen.add(t)
        out.append(p)
        if len(out) >= _MAX_PAIRS:
            break
    return out


# ----------------- Provider shims -----------------

def _search_bing(query: str, *, n: int, freshness: Optional[str], allow: Optional[List[str]]) -> List[str]:
    key = _env("NERION_SEARCH_API_KEY")
    endpoint = _env("NERION_SEARCH_BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
    if not key or not requests:
        logger.debug("[search] Bing unavailable: missing key or requests")
        return []
    headers = {"Ocp-Apim-Subscription-Key": key}
    params = {"q": query, "count": min(max(n, 1), 10)}
    fres = _map_freshness(freshness)
    if fres:
        params["freshness"] = fres
    try:
        NetworkGate.assert_allowed(task_type="web_search", url=endpoint)
        r = requests.get(endpoint, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        urls = [it.get("url") for it in (data.get("webPages", {}) or {}).get("value", []) if it.get("url")]
        urls = [u for u in urls if _allowed(u, allow)]
        return _dedupe_keep_order(urls)
    except Exception as e:  # pragma: no cover
        if isinstance(e, PermissionError):
            raise
        logger.debug(f"[search] Bing error: {e}")
        return []


def _search_serpapi(query: str, *, n: int, freshness: Optional[str], allow: Optional[List[str]]) -> List[str]:
    key = _env("NERION_SEARCH_API_KEY")
    if not key or not requests:
        logger.debug("[search] SerpAPI unavailable: missing key or requests")
        return []
    params = {
        "engine": "google",
        "q": query,
        "api_key": key,
        "num": min(max(n, 1), 10),
        "hl": "en",
        "safe": "active",
    }
    # SerpAPI recency: use tbs=qdr:d/w/m for day/week/month
    fres = _map_freshness(freshness)
    if fres == "Day":
        params["tbs"] = "qdr:d"
    elif fres == "Week":
        params["tbs"] = "qdr:w"
    elif fres == "Month":
        params["tbs"] = "qdr:m"
    try:
        NetworkGate.assert_allowed(task_type="web_search", url="https://serpapi.com/search.json")
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        urls: List[str] = []
        for it in data.get("organic_results", []) or []:
            u = it.get("link")
            if u:
                urls.append(u)
        urls = [u for u in urls if _allowed(u, allow)]
        return _dedupe_keep_order(urls)
    except Exception as e:  # pragma: no cover
        if isinstance(e, PermissionError):
            raise
        logger.debug(f"[search] SerpAPI error: {e}")
        return []


def _search_gcs(query: str, *, n: int, freshness: Optional[str], allow: Optional[List[str]]) -> List[str]:
    key = _env("NERION_SEARCH_API_KEY")
    cx = _env("NERION_SEARCH_GCS_CX")
    if not key or not cx or not requests:
        logger.debug("[search] GCS unavailable: missing key/cx or requests")
        return []
    params = {
        "key": key,
        "cx": cx,
        "q": query,
        "num": min(max(n, 1), 10),
        "safe": "active",
        "hl": "en",
    }
    try:
        NetworkGate.assert_allowed(task_type="web_search", url="https://www.googleapis.com/customsearch/v1")
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        urls = [it.get("link") for it in data.get("items", []) or [] if it.get("link")]
        urls = [u for u in urls if _allowed(u, allow)]
        return _dedupe_keep_order(urls)
    except Exception as e:  # pragma: no cover
        if isinstance(e, PermissionError):
            raise
        logger.debug(f"[search] GCS error: {e}")
        return []


def _search_duck(query: str, *, n: int, freshness: Optional[str], allow: Optional[List[str]]) -> List[str]:
    # Use DuckDuckGo Instant Answer API (not full SERP scraping). It returns limited data;
    # we primarily use it as a last-resort discovery path without violating ToS.
    if not requests:
        return []
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
        "pretty": 0,
    }
    try:
        NetworkGate.assert_allowed(task_type="web_search", url="https://api.duckduckgo.com/")
        r = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        urls: List[str] = []
        # Extract URLs from RelatedTopics and AbstractURL if present
        if data.get("AbstractURL"):
            urls.append(data["AbstractURL"])  # type: ignore
        for topic in data.get("RelatedTopics", []) or []:
            # topics can be nested; handle both dict and dict with "Topics"
            if isinstance(topic, dict) and topic.get("FirstURL"):
                urls.append(topic.get("FirstURL"))
            sub = topic.get("Topics") if isinstance(topic, dict) else None
            if isinstance(sub, list):
                for t in sub:
                    u = t.get("FirstURL") if isinstance(t, dict) else None
                    if u:
                        urls.append(u)
        urls = [u for u in urls if _allowed(u, allow)]
        return _dedupe_keep_order(urls)[:n]
    except Exception as e:  # pragma: no cover
        if isinstance(e, PermissionError):
            raise
        logger.debug(f"[search] DuckDuckGo IA error: {e}")
        return []


_PROVIDER_MAP = {
    "bing": _search_bing,
    "serpapi": _search_serpapi,
    "gcs": _search_gcs,
    "duck": _search_duck,
}


def search_urls(query: str, *, n: int = 5, freshness: Optional[str] = None, allow: Optional[List[str]] = None) -> List[str]:
    """
    Return a de-duplicated list of authoritative URLs for the given query using a compliant Search API provider.

    Provider selection is controlled by env:
      - NERION_SEARCH_PROVIDER = bing | serpapi | gcs | duck
      - NERION_SEARCH_API_KEY = provider API key (if required)
      - NERION_SEARCH_GCS_CX = Google Custom Search CX (if provider = gcs)
    """
    provider = (_env("NERION_SEARCH_PROVIDER", "bing") or "bing").lower()
    fn = _PROVIDER_MAP.get(provider)
    if not fn:
        logger.debug(f"[search] Unknown provider '{provider}', defaulting to bing")
        fn = _search_bing

    try:
        urls = fn(query, n=n, freshness=freshness, allow=allow)
    except Exception as e:  # pragma: no cover
        logger.debug(f"[search] provider call failed: {e}")
        urls = []

    # Fallback provider if nothing returned
    if not urls:
        for alt in ("bing", "serpapi", "gcs", "duck"):
            if alt == provider:
                continue
            fn2 = _PROVIDER_MAP.get(alt)
            if not fn2:
                continue
            try:
                urls = fn2(query, n=n, freshness=freshness, allow=allow)
            except Exception:
                urls = []
            if urls:
                break

    return urls


def _serpapi_search_raw(query: str, *, n: int, freshness: Optional[str], allow: Optional[List[str]]) -> Dict[str, Any]:
    key = _env("NERION_SEARCH_API_KEY")
    if not key or not requests:
        return {"data": None, "urls": []}
    params = {
        "engine": "google",
        "q": query,
        "api_key": key,
        "num": min(max(n, 1), 10),
        "hl": "en",
        "safe": "active",
    }
    fres = _map_freshness(freshness)
    if fres == "Day":
        params["tbs"] = "qdr:d"
    elif fres == "Week":
        params["tbs"] = "qdr:w"
    elif fres == "Month":
        params["tbs"] = "qdr:m"
    try:
        NetworkGate.assert_allowed(task_type="web_search", url="https://serpapi.com/search.json")
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        urls: List[str] = []
        for it in data.get("organic_results", []) or []:
            u = it.get("link")
            if u:
                urls.append(u)
        urls = [u for u in urls if _allowed(u, allow)]
        urls = _dedupe_keep_order(urls)
        return {"data": data, "urls": urls}
    except Exception as e:
        if isinstance(e, PermissionError):
            raise
        return {"data": None, "urls": []}


def search_enriched(query: str, *, n: int = 5, freshness: Optional[str] = None, allow: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Provider-agnostic discovery with optional structured facts extracted generically
    from provider JSON (when available). No topic-specific branching.

    Returns: {"urls": [...], "structured": {"pairs": [...], "sources": [...] } | None}
    """
    provider = (_env("NERION_SEARCH_PROVIDER", "bing") or "bing").lower()

    structured: Optional[Dict[str, Any]] = None
    urls: List[str] = []

    if provider == "serpapi":
        out = _serpapi_search_raw(query, n=n, freshness=freshness, allow=allow)
        urls = out.get("urls", [])
        data = out.get("data")
        if isinstance(data, dict):
            pairs = _flatten_pairs(data)
            # Collect source hosts from organic results
            sources: List[str] = []
            try:
                for it in data.get("organic_results", []) or []:
                    u = it.get("link")
                    cu = _clean_url(u) if u else None
                    if not cu:
                        continue
                    host = urlparse(cu).netloc.replace("www.", "")
                    if host not in sources:
                        sources.append(host)
            except Exception:
                sources = []
            if pairs:
                structured = {"pairs": pairs, "sources": sources[:5]}
    else:
        # Other providers: keep existing URL behavior; no structured block available
        urls = search_urls(query, n=n, freshness=freshness, allow=allow)

    return {"urls": urls, "structured": structured}
