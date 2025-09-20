"""Research utilities for Nerion (consolidated).

This module merges evidence cleaning, snippet fetching, network perf controls,
and the parallel search extraction flow into a single place to reduce module
sprawl. It is topic-agnostic and safe to import from the chat runner.

Exports (stable):
- _has_atomic_numeric, _scrub_boilerplate, _pick_best_sentences, _presynthesize_evidence
- _FETCH_WORKERS, _RENDER_SEM, _get_cached_snippet, _set_cached_snippet
- _fetch_snippet
- run_extraction
"""
from __future__ import annotations

import os
import re
import time
import threading
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import datetime as _dt
from urllib.parse import urlparse as _urlparse

# Site-query engine (used by run_extraction)
from selfcoder.analysis.adapters import engine as _sq_engine

__all__ = [
    "_has_atomic_numeric",
    "_scrub_boilerplate",
    "_pick_best_sentences",
    "_presynthesize_evidence",
    "_FETCH_WORKERS",
    "_RENDER_SEM",
    "_get_cached_snippet",
    "_set_cached_snippet",
    "_fetch_snippet",
    "run_extraction",
]

# ---------------------------------------------------------------------------
# Evidence utilities (from evidence.py)
# ---------------------------------------------------------------------------

def _has_atomic_numeric(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    low = t.lower()
    if '°' in t:
        return True
    if re.search(r"\b\d{1,3}\s*%\b", low):
        return True
    if re.search(r"\b\d{1,4}\s*(?:km|mi|m|s|h|mph|kph|kg|g|lb|oz)\b", low):
        return True
    if re.search(r"\b\d{1,4}([:/.-]\d{1,4})?\b", low):
        return True
    return False

_SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")
_DATE_RX = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?\b|\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.I,
)
_UNITS_RX = re.compile(r"[%°]|\b(?:km|mi|m|s|h|mph|kph|kg|g|lb|oz|usd|eur|gbp|mm|cm|inch|in|ft)\b", re.I)
_SYMBOL_HEAVY_RX = re.compile(r"[^A-Za-z0-9\s,.;:%°/-]")
_DURATION_RX = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_JSONISH_LINE_RX = re.compile(r"[{\[][^\n]+[}\]]|\b(function|document\.|window\.|cookie|var\s+\w+)\b")
_URL_RX = re.compile(r"https?://\S+|\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")


def _scrub_boilerplate(text: str) -> str:
    if not text:
        return ""
    t = _URL_RX.sub("", text)
    lines: List[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _JSONISH_LINE_RX.search(s):
            continue
        if _DURATION_RX.search(s):
            if not _SENT_SPLIT_RX.search(s) or sum(c.isalpha() for c in s) < 8:
                continue
        lines.append(s)
    t = " ".join(lines)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _pick_best_sentences(text: str, k: int = 2) -> str:
    if not text:
        return ""
    parts = _SENT_SPLIT_RX.split(text)
    scored = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if not (5 <= len(words) <= 40):
            continue
        if len(_SYMBOL_HEAVY_RX.findall(s)) > 3:
            continue
        score = 0
        if _DATE_RX.search(s):
            score += 2
        if _UNITS_RX.search(s) or re.search(r"\d", s):
            score += 1
        letters = [c for c in s if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) > 0.7 * len(letters):
            score -= 1
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[: max(1, k)]]
    return " ".join(top)


def _extract_date_ts(text: str) -> Optional[int]:
    """Extract a rough publication date from text and return epoch seconds.
    Supports common patterns without heavy deps; best-effort only.
    """
    try:
        if not text:
            return None
        m = _DATE_RX.search(text)
        if not m:
            return None
        s = m.group(0)
        s = s.strip()
        # Try several simple parses
        fmts = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%b %d, %Y", "%b %d"]
        for f in fmts:
            try:
                dt = _dt.datetime.strptime(s, f)
                # If year absent, assume current year
                if f == "%b %d":
                    dt = dt.replace(year=_dt.datetime.now().year)
                return int(dt.timestamp())
            except Exception:
                continue
        return None
    except Exception:
        return None


def _presynthesize_evidence(artifacts: list, *, max_items: int = 4, max_snip: int = 600) -> str:
    if not artifacts:
        return "No evidence available."

    def _clean(text: str) -> str:
        if not text:
            return ""
        t = str(text).strip()
        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    lines: List[str] = []
    seen_urls = set()
    for a in artifacts[:max_items]:
        try:
            url = (a or {}).get("url", "")
        except Exception:
            url = ""
        if url in seen_urls:
            continue
        seen_urls.add(url)

        head = _clean((a or {}).get("headline", "")) or None
        win = _clean((a or {}).get("winner", "")) or None
        rec = _clean((a or {}).get("recommendation", "")) or None
        raw_snip = (a or {}).get("snippet") or ""
        snip_clean = _scrub_boilerplate(_clean(raw_snip))
        snip = _pick_best_sentences(snip_clean, k=2)[:max_snip] if snip_clean else ""
        lead = head or win or rec or ""
        domains = (a or {}).get("domains") or []
        dom_part = ""
        if domains:
            doms = [d.replace("www.", "").strip() for d in domains if d]
            if doms:
                dom_part = f" ({'; '.join(doms[:2])})"
        if lead and snip:
            lines.append(f"- {lead}{dom_part}\n  {snip}")
        elif lead:
            lines.append(f"- {lead}{dom_part}")
        elif snip:
            lines.append(f"- {snip}{dom_part}")
        else:
            try:
                host = _urlparse(url).netloc.replace("www.", "")
            except Exception:
                host = ""
            if host:
                lines.append(f"- Source: {host}")
    if not lines:
        return "No evidence available."
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Net perf & cache (from netperf.py)
# ---------------------------------------------------------------------------
try:
    from selfcoder.config import get_policy as _get_policy
except Exception:
    def _get_policy():
        return 'balanced'

_POL = _get_policy()
_FETCH_WORKERS = int(os.getenv('NERION_FETCH_WORKERS', '3' if _POL == 'safe' else ('6' if _POL == 'fast' else '4')))
_RENDER_MAX = int(os.getenv('NERION_MAX_CONCURRENT_RENDER', '1' if _POL == 'safe' else ('3' if _POL == 'fast' else '2')))
_RENDER_SEM = threading.Semaphore(_RENDER_MAX)
_SNIPPET_TTL_S = int(os.getenv('NERION_SNIPPET_TTL', '600'))  # 10 minutes default
_SNIPPET_CACHE: dict[Tuple[str, int], tuple[str, float]] = {}

# Basic paywall skip list (hard paywalls)
_PAYWALL_HOSTS = {
    'wsj.com', 'www.wsj.com',
    'ft.com', 'www.ft.com',
    'bloomberg.com', 'www.bloomberg.com',
}

def _norm_host(host: str) -> str:
    h = (host or '').lower()
    h = h.replace('www.', '').replace('m.', '').replace('amp.', '')
    return h


def _cache_key(url: str, query: str) -> tuple[str, int]:
    return (url or ''), hash(query or '')


def _get_cached_snippet(url: str, query: str) -> str:
    try:
        k = _cache_key(url, query)
        snip, ts = _SNIPPET_CACHE.get(k, (None, 0))
        if snip and (time.time() - ts) <= _SNIPPET_TTL_S:
            return snip  # type: ignore[return-value]
    except Exception:
        pass
    return ''


def _set_cached_snippet(url: str, query: str, snippet: str) -> None:
    try:
        if snippet:
            _SNIPPET_CACHE[_cache_key(url, query)] = (snippet, time.time())
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Snippet fetcher (from snippets.py)
# ---------------------------------------------------------------------------

def _fetch_snippet(url: str, query: str, max_chars: int = 1200) -> tuple[str, int]:
    """Fetch plain text for a URL and return a clean, short snippet.
    Strategy:
      1) Non-rendered fetch → scrub → pick best sentences → return if informative.
      2) Fallback to rendered fetch (guarded by render semaphore) → scrub → pick.
    """
    try:
        from selfcoder.analysis import docs as _docs  # local import to avoid module-scope coupling
    except Exception:
        return ('', 0)
    try:
        d = _docs.read_doc(path=None, url=url, query=query, timeout=(12 if _POL=='fast' else 20), render=False)
        raw = (d.get('raw_text') or d.get('text') or '').strip()
        if raw:
            cleaned = _scrub_boilerplate(raw)
            best = _pick_best_sentences(cleaned, k=2)
            if best and (_has_atomic_numeric(best) or len(best) >= 40):
                return (best[:max_chars], int(d.get('cache_age_s') or 0))
        # Slow path: rendered (guarded)
        with _RENDER_SEM:
            d2 = _docs.read_doc(path=None, url=url, query=query, timeout=(15 if _POL=='fast' else 25), render=True)
        raw2 = (d2.get('raw_text') or d2.get('text') or '').strip()
        if raw2:
            cleaned2 = _scrub_boilerplate(raw2)
            best2 = _pick_best_sentences(cleaned2, k=2)
            if best2:
                return (best2[:max_chars], int(d2.get('cache_age_s') or 0))
    except Exception:
        pass
    return ('', 0)

# ---------------------------------------------------------------------------
# Search extraction flow (from search_flow.py)
# ---------------------------------------------------------------------------

def _extract_from_single_url(u: str, query: str) -> Optional[Dict[str, Any]]:
    try:
        host = _urlparse(u).netloc
        profile_key = f"auto:{host}"
        out = _sq_engine.run(profile_key, query=query, url=u)
        res = out.get('result', {}) or {}
        cons = res.get('consensus', {}) or {}
        cite_domains: List[str] = []
        try:
            for cit in (out.get('citations') or []):
                if (cit or {}).get('source') == 'external':
                    h = _urlparse((cit or {}).get('url') or '').netloc
                    if h and h not in cite_domains:
                        cite_domains.append(h.replace('www.', ''))
        except Exception:
            pass
        head = res.get('headline') or res.get('conclusion') or ''
        rec_txt = res.get('recommendation') or ''
        win_name = (cons.get('winner') or {}).get('name') or ''
        snippet = ''
        cache_age_s = 0
        structured_text = ' '.join([x for x in (head, rec_txt, win_name) if x])
        # Fetch a snippet when we lack numeric evidence OR the visible text looks like a date-only header
        if (not _has_atomic_numeric(structured_text)) or _DATE_RX.search(structured_text or ''):
            cached = _get_cached_snippet(u, query)
            if cached:
                snippet = cached
            else:
                sn, age = _fetch_snippet(u, query, max_chars=1200)
                if sn:
                    snippet = sn
                    cache_age_s = age
                    _set_cached_snippet(u, query, sn)
        # Approximate recency from snippet/head if present
        date_ts = _extract_date_ts(' '.join([head, rec_txt, snippet]))
        return {
            'url': u,
            'headline': head,
            'recommendation': rec_txt,
            'winner': win_name,
            'confidence': res.get('confidence'),
            'domains': cite_domains,
            'snippet': snippet,
            'cache_age_s': cache_age_s,
            'date_ts': date_ts,
        }
    except Exception:
        return None


def _prepend_structured_artifact(artifacts: List[Dict[str, Any]], structured: Optional[Dict[str, Any]]) -> None:
    try:
        if structured and structured.get('pairs'):
            pairs = structured['pairs']
            domains = structured.get('sources', [])
            sn_lines: List[str] = []
            for p in pairs:
                v = (p or {}).get('value') or ''
                if _has_atomic_numeric(v):
                    k = (p or {}).get('key') or ''
                    sn_lines.append(f"- {k}: {v}")
            art = {
                'url': '',
                'headline': '',
                'recommendation': '',
                'winner': '',
                'confidence': None,
                'domains': domains,
                'snippet': "\n".join(sn_lines) if sn_lines else '',
            }
            artifacts.insert(0, art)
    except Exception:
        pass


def run_extraction(urls: List[str], query: str, structured: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], str]:
    """Run the full Analyst stage and return (artifacts, snippet_block).

    Steps:
      1) Parallel per-URL extraction (headline/reco/winner + cached/best snippet)
      2) Final fallback: fetch plain text for top URLs if artifacts are blank
      3) Prepend structured facts (numeric pairs) as an artifact, if present
      4) Build snippet_block; if it lacks numeric evidence, enrich with top URL snippets and rebuild
    """
    artifacts: List[Dict[str, Any]] = []

    # Pre-filter URLs: skip hard paywalls and de-duplicate by normalized host
    dedup: List[str] = []
    seen_hosts: set[str] = set()
    for u in urls or []:
        try:
            host = _norm_host(_urlparse(u).netloc)
        except Exception:
            host = ''
        if not host:
            continue
        if host in _PAYWALL_HOSTS:
            continue
        if host in seen_hosts:
            continue
        seen_hosts.add(host)
        dedup.append(u)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as ex:
        futs = [ex.submit(_extract_from_single_url, u, query) for u in dedup[:4]]
        for fut in concurrent.futures.as_completed(futs):
            art = fut.result()
            if art:
                artifacts.append(art)

    if (not artifacts) or all(
        not (a.get('headline') or a.get('recommendation') or a.get('winner') or a.get('snippet'))
        for a in artifacts
    ):
        fallback_snips: List[Dict[str, str]] = []
        for u in dedup[:2]:
            try:
                sn = _get_cached_snippet(u, query) or _fetch_snippet(u, query, max_chars=1200)
                if sn:
                    _set_cached_snippet(u, query, sn)
                    fallback_snips.append({'url': u, 'snippet': sn})
            except Exception:
                continue
        if fallback_snips:
            artifacts = [{
                'url': fs['url'],
                'headline': '',
                'recommendation': '',
                'winner': '',
                'confidence': None,
                'domains': [],
                'snippet': fs['snippet'],
            } for fs in fallback_snips]

    _prepend_structured_artifact(artifacts, structured)

    # Prefer more recent evidence when available and score by freshness, trust, agreement
    try:
        def _trust(host: str) -> float:
            h = (host or '').lower()
            if h.endswith('.gov') or h.endswith('.edu'):
                return 1.0
            if 'docs.python.org' in h or 'github.com' in h:
                return 0.9
            if h.endswith('.org'):
                return 0.7
            return 0.5
        # Agreement: count common numeric tokens
        from collections import Counter
        nums = []
        for a in artifacts:
            nums += re.findall(r"\d+(?:[.,]\d+)?", a.get('snippet') or '')
        num_counts = Counter(nums)
        now = int(time.time())
        def _score(a):
            # Freshness: newer is better
            dt = int(a.get('date_ts') or 0)
            freshness = 0.0
            if dt:
                age = max(1, now - dt)
                freshness = max(0.0, 1.0 - (age / (90*86400)))  # 90d horizon
            # Trust: by domain
            try:
                host = _urlparse(a.get('url') or '').netloc
            except Exception:
                host = ''
            trust = _trust(host)
            # Agreement: reward common numbers
            agree = 0.0
            try:
                vals = set(re.findall(r"\d+(?:[.,]\d+)?", a.get('snippet') or ''))
                agree = sum(num_counts.get(v, 0) for v in vals) / (len(nums) or 1)
            except Exception:
                agree = 0.0
            return (freshness*0.5 + trust*0.3 + agree*0.2)
        artifacts.sort(key=_score, reverse=True)
    except Exception:
        pass

    snippet_block = _presynthesize_evidence(artifacts)

    if not _has_atomic_numeric(snippet_block):
        for u in (dedup[:2] if dedup else []):
            if any((a.get('url') == u and (a.get('snippet') or '').strip()) for a in artifacts):
                continue
            sn = _get_cached_snippet(u, query) or _fetch_snippet(u, query, max_chars=1200)
            if sn:
                _set_cached_snippet(u, query, sn)
                artifacts.append({
                    'url': u,
                    'headline': '',
                    'recommendation': '',
                    'winner': '',
                    'confidence': None,
                    'domains': [],
                    'snippet': sn,
                })
        snippet_block = _presynthesize_evidence(artifacts)

    # Simple contradiction flagging: same sentence pattern with different numbers
    try:
        import collections
        patt_values: Dict[str, set] = collections.defaultdict(set)
        def _normalize_claim(s: str) -> str:
            # Drop numbers to form a loose pattern
            return re.sub(r"\d+(?:[.,]\d+)?", "<num>", s.lower())
        for a in artifacts:
            sn = (a.get('snippet') or '')
            for sent in _SENT_SPLIT_RX.split(sn):
                sent = sent.strip()
                if not sent:
                    continue
                if not _has_atomic_numeric(sent):
                    continue
                patt = _normalize_claim(sent)
                nums = re.findall(r"\d+(?:[.,]\d+)?", sent)
                for n in nums[:2]:
                    patt_values[patt].add(n)
        contrad = [p for p, vals in patt_values.items() if len(vals) >= 2]
        if contrad:
            snippet_block = snippet_block + "\n" + "\n".join([f"Contradiction flagged: pattern '{p[:80]}'" for p in contrad[:2]])
    except Exception:
        pass

    return artifacts, snippet_block
