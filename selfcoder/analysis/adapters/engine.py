from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urlparse

import time
from html.parser import HTMLParser
from urllib.parse import urljoin

from urllib.parse import urldefrag

import re
from selfcoder.analysis import docs as docs_mod
from selfcoder.analysis.adapters import store as store_mod
from selfcoder.config import allow_network  # Defensive network gate
from selfcoder.analysis.knowledge import index as kb_index  # Knowledge diff/learning calibration
from selfcoder.analysis.knowledge.diff import diff_artifacts, adjust_weights, track_calibration, text_hash
from selfcoder.analysis.consensus.aggregate import aggregate_consensus  # Consensus aggregator

def _first_sentence(text: str, max_len: int = 240) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    try:
        m = re.search(r"[.!?]\s", t)
    except Exception:
        m = None
    s = t if not m else t[:m.end()].strip()
    return (s[:max_len] + "…") if len(s) > max_len else s

_BOILERPLATE_ANCHOR_WORDS = {
    "skip", "footer", "language", "country", "region", "sitemap", "privacy", "terms",
}

_RECO_KWS = {"best", "top", "featured", "recommended", "series", "laptop", "notebook", "desktop", "all-in-one", "aio", "printer", "laser", "inkjet"}

_MODEL_HINTS = {"pc", "laptop", "notebook", "desktop", "all-in-one", "aio", "printer", "laser", "inkjet", "book", "series", "ultra", "pro"}

# Stoplist for model extraction and best-of template helper
_STOP_TOKENS = {
    "home", "product", "products", "shop", "category", "categories", "terms", "privacy",
    "support", "faq", "program", "offers", "offer", "savings", "subscribe", "subscription"
}


# Headlines that start like these are often marketing/FAQ headers; prefer winner-based line
_HEADLINE_BOILERPLATE_PREFIXES = (
    "what’s", "whats", "what is", "make every", "learn more", "compare", "explore",
)

# Normalize a headline for robust prefix checks
def _norm_head(s: str) -> str:
    s = (s or "").strip().lower()
    # normalize curly quotes and unicode variants to ascii for robust prefix checks
    return s.replace("’", "'").replace("“", '"').replace("”", '"')

def _best_template(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(value|budget|price|save|savings)\b", t):
        return "best value pick"
    if re.search(r"\b(pro|business|enterprise|elite)\b", t):
        return "best for pros"
    return "best overall"

def _extract_models(text: str, max_items: int = 6) -> list[str]:
    """Heuristic model extractor: finds short title-cased spans containing model hints.
    Returns a de-duplicated list preserving order.
    """
    if not text:
        return []
    spans = re.findall(r"\b([A-Z][A-Za-z0-9]+(?: [A-Z][A-Za-z0-9]+){0,3})\b", text)
    out: list[str] = []
    seen = set()
    for s in spans:
        low = s.lower()
        if any(st in low for st in _STOP_TOKENS):
            continue
        if any(h in low for h in _MODEL_HINTS):
            key = re.sub(r"\s+", " ", s.strip())
            if key and key.lower() not in _STOP_TOKENS and key not in seen:
                seen.add(key)
                out.append(key)
            if len(out) >= max_items:
                break
    return out

def _name_from_url(u: str) -> str:
    try:
        parsed = urlparse(u)
        p = parsed.path or ""
    except Exception:
        p = ""
    # take last non-empty segment
    seg = ""
    for part in p.split("/"):
        if part:
            seg = part
    # strip common extensions
    seg = re.sub(r"\.(html?|php|aspx)$", "", seg, flags=re.IGNORECASE)
    # replace separators
    seg = seg.replace("-", " ").replace("_", " ")
    seg = re.sub(r"\s+", " ", seg).strip()
    # keep only letters/numbers/spaces
    seg = re.sub(r"[^0-9A-Za-z ]+", "", seg)
    if not seg:
        return ""
    # Title-case then fix common acronyms
    words = seg.title().split()
    acronyms = {"ai", "pc", "vr", "xr", "ml", "nlp", "gpu", "cpu", "ssd", "oled"}

    def _fix_acronym(token: str) -> str:
        lw = token.lower()
        if lw in acronyms:
            return lw.upper()
        if lw.endswith("s") and lw[:-1] in acronyms:
            return lw[:-1].upper() + "s"
        return token

    fixed = [_fix_acronym(w) for w in words]
    return " ".join(fixed)


def _norm_url(u: str) -> str:
    # strip fragments and trim whitespace
    try:
        clean, _ = urldefrag(u.strip())
        return clean
    except Exception:
        return u.strip()


def _is_fragment_only(href: str) -> bool:
    return href.strip().startswith("#")
 

# -----------------------------
# Generic, domain‑agnostic adapter engine
# Uses profile policies (depth, max_pages, render, timeouts) but
# keeps logic neutral — no hard‑coded industries.
# -----------------------------


def _same_host(a: str, b: str) -> bool:
    try:
        ua = urlparse(a).netloc.split(":")[0].lower()
        ub = urlparse(b).netloc.split(":")[0].lower()
        return bool(ua and ub) and (ua == ub)
    except Exception:
        return False


def _keyword_overlap(query: str, text: str) -> float:
    q = {w for w in (query or "").lower().split() if len(w) >= 3}
    if not q:
        return 0.0
    t = set(w for w in (text or "").lower().split())
    if not t:
        return 0.0
    inter = len(q & t)
    return inter / max(len(q), 1)


class _LinkExtractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base = base_url
        self.links: list[tuple[str, str]] = []  # (url, anchor_text)
        self._in_a = False
        self._anchor_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._in_a = True
            href = None
            for k, v in attrs:
                if k.lower() == "href":
                    href = v
                    break
            if href:
                abs_url = urljoin(self.base, href)
                self.links.append((abs_url, ""))

    def handle_data(self, data):
        if self._in_a and data:
            self._anchor_buf.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a":
            # attach last collected anchor text to last link if present
            if self.links:
                url, _ = self.links[-1]
                anchor = " ".join(self._anchor_buf).strip()
                self.links[-1] = (url, anchor)
            self._in_a = False
            self._anchor_buf = []


def _fetch_html(url: str, *, timeout: int, render: bool, render_timeout: int) -> str:
    """Best-effort fetch of raw HTML for link discovery.
    - If render=True, use playwright (docs-web extra) to render HTML.
    - Else, try requests; if unavailable, return empty string (degrades to single-page gather).
    """
    # Global safety: respect network gate in library code too
    try:
        if not allow_network():
            return ""
    except Exception:
        # Fail-closed to empty string if config check errors
        return ""
    try:
        if render:
            from selfcoder.analysis import web_render
            return web_render.render_url(url, timeout=timeout, render_timeout=render_timeout)
        else:
            try:
                import requests  # type: ignore
            except Exception:
                return ""
            headers = {"User-Agent": "nerion-adapt/1.0"}
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
    except Exception:
        return ""


def _extract_links_from_html(html: str, base_url: str) -> list[tuple[str, str]]:
    if not html:
        return []
    parser = _LinkExtractor(base_url)
    try:
        parser.feed(html)
    except Exception:
        return []
    # de-dup while preserving order
    seen = set()
    out: list[tuple[str, str]] = []
    for u, a in parser.links:
        key = (u.strip(), a.strip())
        if key[0] and key not in seen:
            seen.add(key)
            out.append((key[0], key[1]))
    return out


def gather(profile: Dict[str, Any], *, query: str, url: str) -> List[Dict[str, Any]]:
    """Collect initial evidence using the existing docs pipeline.

    MVP: fetch the root URL only (bounded gather). Profile policies are read and
    used for render/timeout; future iterations may crawl depth>1 and multiple pages.
    """
    pol = profile.get("source_policies", {})
    render = bool(pol.get("render", False))
    timeout = int(pol.get("timeout", 10))
    render_timeout = int(pol.get("render_timeout", 5))
    depth = int(pol.get("depth", 1))
    user_max = int(pol.get("max_pages", 6))

    content_hints = set(map(str.lower, pol.get("content_hints", [
        "product", "products", "shop", "category", "catalog", "laptop", "notebook", "pc", "support", "news"
    ])))
    # Expand content hints from the query to steer navigation (prevents laptop bias on printer/desktop queries)
    qh = set()
    qlow = (query or "").lower()
    if any(k in qlow for k in ("printer", "laser", "inkjet")):
        qh.update(["printer", "printers", "laser", "inkjet", "all-in-one", "aio"])
    if any(k in qlow for k in ("desktop", "tower", "all-in-one", "aio")):
        qh.update(["desktop", "desktops", "tower", "all-in-one", "aio"])
    if any(k in qlow for k in ("laptop", "notebook")):
        qh.update(["laptop", "laptops", "notebook", "notebooks"]) 
    if qh:
        content_hints |= qh
    query_tokens = {w for w in (query or "").lower().split() if len(w) >= 3}

    # Adaptive caps
    effective_max = min(user_max, 4 if render else 8)
    TIME_BUDGET = 45.0  # seconds
    BYTES_BUDGET = 8 * 1024 * 1024  # 8 MB

    start_ts = time.time()
    bytes_used = 0

    evidence: List[Dict[str, Any]] = []
    visited: set[str] = set()
    queue: list[tuple[str, int, str]] = [(url, 0, "")]  # (url, depth_level, anchor_text)

    # Per-run memo to avoid refetching the same page
    _fetch_cache: Dict[str, str] = {}
    def _get_html(u: str) -> str:
        u_n = _norm_url(u)
        cached = _fetch_cache.get(u_n)
        if cached is not None:
            return cached
        html = _fetch_html(u_n, timeout=timeout, render=render, render_timeout=render_timeout)
        _fetch_cache[u_n] = html
        return html

    def budgets_ok() -> bool:
        if (time.time() - start_ts) > TIME_BUDGET:
            return False
        if bytes_used > BYTES_BUDGET:
            return False
        return True

    consecutive_low = 0
    fallback_used = False

    while queue and len(evidence) < effective_max and budgets_ok():
        cur_url, lvl, anchor_txt = queue.pop(0)
        cur_url = _norm_url(cur_url)
        if cur_url in visited:
            continue
        visited.add(cur_url)

        # Fetch page using docs pipeline to keep classification & normalization consistent
        doc = docs_mod.read_doc(
            path=None,
            url=cur_url,
            query=query,
            timeout=timeout,
            render=render,
            render_timeout=render_timeout,
            selector=None,
        )

        rf = bool(doc.get("render_fallback"))

        text = (doc.get("raw_text") or doc.get("text", ""))
        bytes_used += len(text.encode("utf-8", errors="ignore"))

        evidence.append(
            {
                "url": _norm_url(doc.get("url") or doc.get("path") or ""),
                "source": doc.get("source"),
                "text": text,
                "domain": doc.get("domain"),
                "domain_confidence": doc.get("domain_confidence"),
                "render_fallback": rf,
            }
        )

        # Early-stop on low-utility streak
        ov = _keyword_overlap(query, text)
        if ov < 0.05:
            consecutive_low += 1
        else:
            consecutive_low = 0
        if consecutive_low >= 3:
            if not fallback_used and lvl <= depth:
                # one-time fallback: pick longest-anchor same-domain link to try to jump to a content section
                html = _get_html(cur_url)
                links = _extract_links_from_html(html, cur_url)
                cand: list[tuple[int, tuple[str, str]]] = []
                for href, anchor in links:
                    if _is_fragment_only(href):
                        continue
                    href_nf = _norm_url(href)
                    if not href_nf.lower().startswith("http"):
                        continue
                    if not _same_host(url, href_nf):
                        continue
                    if href_nf in visited:
                        continue
                    a = (anchor or "").strip()
                    if any(bp in a.lower() for bp in _BOILERPLATE_ANCHOR_WORDS):
                        continue
                    cand.append((len(a), (href_nf, anchor)))
                cand.sort(key=lambda x: x[0], reverse=True)
                if cand:
                    href_nf, anchor = cand[0][1]
                    queue.append((href_nf, lvl + 1, anchor))
                    fallback_used = True
                    continue
            break

        # Enqueue same-domain links if within depth
        if lvl < depth:
            html = _get_html(cur_url)
            links = _extract_links_from_html(html, cur_url)
            # score links by anchor/query overlap, highest first
            scored: list[tuple[float, tuple[str, str]]] = []
            for href, anchor in links:
                if _is_fragment_only(href):
                    continue
                href_nf = _norm_url(href)
                if not href_nf.lower().startswith("http"):
                    continue
                if not _same_host(url, href_nf):
                    continue
                if href_nf in visited:
                    continue
                a = (anchor or "").strip().lower()
                if any(bp in a for bp in _BOILERPLATE_ANCHOR_WORDS):
                    continue
                # base score: anchor/query overlap
                overlap = _keyword_overlap(query, a)
                # path/content-hint bonus
                path = urlparse(href_nf).path.lower()
                path_bonus = 0.0
                if query_tokens and any(t in path for t in query_tokens):
                    path_bonus += 0.15
                if content_hints and any(h in path for h in content_hints):
                    path_bonus += 0.10
                link_score = overlap + min(path_bonus, 0.25)
                scored.append((link_score, (href_nf, anchor)))
            scored.sort(key=lambda x: x[0], reverse=True)
            queued_set = { _norm_url(u) for (u, _, _) in queue }
            for _, (href_nf, anchor) in scored:
                if len(queue) + len(evidence) >= effective_max:
                    break
                if href_nf in queued_set or href_nf in visited:
                    continue
                queue.append((href_nf, lvl + 1, anchor))

    return evidence


def rank(profile: Dict[str, Any], *, query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score evidence items with generic, transparent heuristics.

    Features:
      - keyword_overlap(query, text)
      - onsite_signal (same host vs external)
      - coverage (text length proxy)
    Weights come from profile.scoring.
    """
    scoring = profile.get("scoring", {})
    w_recency = float(scoring.get("recency", 0.30))  # placeholder (no recency yet)
    w_authority = float(scoring.get("authority", 0.25))  # placeholder (no authority yet)
    w_agreement = float(scoring.get("agreement", 0.20))  # placeholder (no agreement yet)
    w_onsite = float(scoring.get("onsite_signal", 0.15))
    w_coverage = float(scoring.get("coverage", 0.10))

    root_url = None
    for e in evidence:
        if e.get("url"):
            root_url = e["url"]
            break

    ranked: List[Dict[str, Any]] = []
    for ev in evidence:
        text = ev.get("text", "")
        overlap = _keyword_overlap(query, text)
        onsite = 1.0 if (root_url and ev.get("url") and _same_host(root_url, ev["url"])) else 0.0
        coverage = min(len(text) / 4000.0, 1.0)  # cap coverage contribution

        # Currently recency/authority/agreement are placeholders (0); keep weights for compatibility.
        score = (
            w_onsite * onsite
            + w_coverage * coverage
            + w_authority * 0.0
            + w_recency * 0.0
            + w_agreement * 0.0
            + 0.5 * overlap  # direct boost for overlap (transparent)
        )

        ranked.append(
            {
                **ev,
                "score": float(score),
                "features": {
                    "overlap": overlap,
                    "onsite": onsite,
                    "coverage": coverage,
                },
                "rationale": [
                    f"overlap={overlap:.2f}",
                    f"onsite={'yes' if onsite else 'no'}",
                    f"coverage~{coverage:.2f}",
                ],
            }
        )

    ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked


def synthesize(profile: Dict[str, Any], *, query: str, ranked: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce a compact summary from top evidence using the docs summarizer.

    Returns a dict containing: summary, bullets, conclusion, confidence, citations.
    """
    top_text = (ranked[0].get("text") if ranked else "") or ""
    # Use multiple top texts (up to 3) for a richer summary
    joined_text = top_text
    for ev in ranked[1:3]:
        joined_text += ("\n" + (ev.get("text") or ""))
    summary_obj = docs_mod.summarize_text(joined_text)

    # Heuristic signals for confidence tuning
    combined_low = joined_text.lower()
    has_compare_kw = bool(re.search(r"\b(best|recommend|flagship|compare|vs)\b", combined_low))
    has_marketing_kw = bool(re.search(r"\b(buy now|add to cart|select country|shop now)\b", combined_low))

    # Confidence blending with simple heuristics
    top_score = ranked[0].get("score", 0.0) if ranked else 0.0
    conf = float(top_score)
    if has_compare_kw:
        conf += 0.08
    if has_marketing_kw:
        conf -= 0.05
    confidence = max(0.0, min(1.0, conf))

    # Source breakdown (onsite/external) from ranked evidence
    source_counts = {"onsite": 0, "external": 0}
    for ev in ranked:
        src_raw = (ev.get("source") or "").lower()
        if src_raw == "external":
            source_counts["external"] += 1
        else:
            source_counts["onsite"] += 1

    citations: List[Dict[str, Any]] = []
    try:
        sorted_by_overlap = sorted(
            ranked,
            key=lambda ev: (float(ev.get("features", {}).get("overlap", 0.0)), float(ev.get("score", 0.0))),
            reverse=True,
        )
        primary = [ev for ev in sorted_by_overlap if ev.get("url") and ev.get("features", {}).get("overlap", 0.0) > 0.0]
        secondary = [ev for ev in sorted_by_overlap if ev.get("url") and ev.get("features", {}).get("overlap", 0.0) == 0.0]
        chosen = primary[:5] if len(primary) >= 3 else (primary + secondary)[:5]
        for ev in chosen:
            src_raw = (ev.get("source") or "").lower()
            if not src_raw and ev.get("features", {}).get("onsite", 0.0):
                src = "onsite"
            elif src_raw == "url":
                src = "onsite"
            elif src_raw == "external":
                src = "external"
            else:
                src = src_raw
            cit = {
                "url": ev["url"],
                "why": "; ".join(ev.get("rationale", [])),
                "source": src,
                "why_obj": {
                    "overlap": float(ev.get("features", {}).get("overlap", 0.0)),
                    "onsite": bool(ev.get("features", {}).get("onsite", 0.0)),
                    "coverage": float(ev.get("features", {}).get("coverage", 0.0)),
                    "score": float(ev.get("score", 0.0)),
                },
            }
            if src == "external":
                dh = ev.get("date_hint")
                if dh:
                    cit["date_hint"] = dh
            citations.append(cit)
    except Exception:
        for ev in ranked[:5]:
            if ev.get("url"):
                src_raw = (ev.get("source") or "").lower()
                if not src_raw and ev.get("features", {}).get("onsite", 0.0):
                    src = "onsite"
                elif src_raw == "url":
                    src = "onsite"
                elif src_raw == "external":
                    src = "external"
                else:
                    src = src_raw
                cit = {
                    "url": ev["url"],
                    "why": "; ".join(ev.get("rationale", [])),
                    "source": src,
                    "why_obj": {
                        "overlap": float(ev.get("features", {}).get("overlap", 0.0)),
                        "onsite": bool(ev.get("features", {}).get("onsite", 0.0)),
                        "coverage": float(ev.get("features", {}).get("coverage", 0.0)),
                        "score": float(ev.get("score", 0.0)),
                    },
                }
                if src == "external":
                    dh = ev.get("date_hint")
                    if dh:
                        cit["date_hint"] = dh
                citations.append(cit)

    # Simple onsite vs external agreement adjustment for confidence
    try:
        onsite_ov = [float(ev.get("features", {}).get("overlap", 0.0)) for ev in ranked if ev.get("source") != "external"]
        external_ov = [float(ev.get("features", {}).get("overlap", 0.0)) for ev in ranked if ev.get("source") == "external"]
        # Average overlaps over top few items
        def _avg(xs):
            xs = xs[:3]
            return (sum(xs) / len(xs)) if xs else 0.0
        avg_on = _avg(onsite_ov)
        avg_ex = _avg(external_ov)
        if avg_on > 0.15 and avg_ex > 0.10:
            confidence = min(1.0, confidence + 0.05)  # corroboration
        elif avg_on > 0.20 and avg_ex < 0.02:
            confidence = max(0.0, confidence - 0.05)  # weak external support
    except Exception:
        pass

    # Generate model bullets if summarizer returns none, with filtering
    model_bullets = [
        m for m in _extract_models(joined_text)
        if len(m.split()) >= 2 and not m.lower().startswith(("pc", "ai")) and not any(st in m.lower() for st in _STOP_TOKENS)
    ]
    bullets = summary_obj.get("bullets", []) or model_bullets[:4]

    # Lightweight recommendation from top evidence (enriched, researched phrasing)
    recommendation = ""
    headline = ""
    try:
        # Domain terms derived from the user query to guide candidate selection
        q_terms = set(w for w in (query or "").lower().split())
        want_printer = any(w in q_terms for w in {"printer", "printers", "laser", "inkjet"})
        want_desktop = any(w in q_terms for w in {"desktop", "desktops", "tower", "all-in-one", "aio"})
        want_laptop = any(w in q_terms for w in {"laptop", "laptops", "notebook", "notebooks"})
        picks = ranked[:3]
        cand: list[tuple[float, str]] = []  # (score, name)
        for ev in picks:
            u = ev.get("url", "")
            name = _name_from_url(u)
            text_low = (ev.get("text") or "").lower()
            path_low = urlparse(u).path.lower() if u else ""
            has_kw = any(k in text_low or k in path_low for k in _RECO_KWS)
            # Penalize candidates that conflict with the requested domain (printer/desktop/laptop)
            penalty = 0.0
            if want_printer:
                if not ("printer" in text_low or "printer" in path_low or "laser" in text_low or "inkjet" in text_low):
                    penalty -= 0.25
            if want_desktop:
                if ("laptop" in text_low or "notebook" in text_low or "laptop" in path_low or "notebook" in path_low):
                    penalty -= 0.30
                if not ("desktop" in text_low or "all-in-one" in text_low or "aio" in text_low or "tower" in text_low or "desktop" in path_low):
                    penalty -= 0.15
            if want_laptop:
                if not ("laptop" in text_low or "notebook" in text_low or "laptop" in path_low or "notebook" in path_low):
                    penalty -= 0.20
            score = float(ev.get("score", 0.0)) + (0.1 if has_kw else 0.0) + (0.05 if name else 0.0) + penalty
            if name or has_kw:
                cand.append((score, name or u))
        cand.sort(key=lambda x: x[0], reverse=True)
        if cand:
            primary = cand[0][1]
            alts = [c[1] for c in cand[1:] if c[1]]
            if primary:
                tpl = _best_template(combined_low)
                # Educated, researched phrasing
                rec_bits = []
                if has_compare_kw:
                    rec_bits.append("emphasizes comparison/positioning")
                # lightweight quality hints
                if re.search(r"battery|performance|portable|lightweight|display", combined_low):
                    rec_bits.append("highlights quality signals (battery/performance/portability/display)")
                detail = "; ".join(rec_bits) if rec_bits else "is prominently positioned on-site"
                recommendation = (
                    f"According to the site, {primary} is presented as the {tpl} for '{query}', {detail}."
                )
                if alts:
                    recommendation += f" Alternatives highlighted include {', '.join(alts)}."

        # headline: concise version for UI
        if cand:
            primary = cand[0][1]
            alts = [c[1] for c in cand[1:] if c[1]]
            if primary:
                tpl = _best_template(combined_low)
                if alts:
                    headline = f"For '{query}', {primary} appears to be the {tpl}. Alternatives: {', '.join(alts)}."
                else:
                    headline = f"For '{query}', {primary} appears to be the {tpl}."
        # If the chosen candidate contradicts the requested domain, clear it so the consensus text or summary is used
        if cand:
            chosen_name = cand[0][1]
            cn_low = (chosen_name or "").lower()
            if want_printer and not any(k in cn_low for k in ("printer", "laser", "inkjet")):
                headline = headline if "printer" in (headline or "").lower() else ""
            if want_desktop and any(k in cn_low for k in ("laptop", "notebook")):
                headline = headline if "desktop" in (headline or "").lower() else ""
    except Exception:
        recommendation = ""
        headline = ""

    # Distinct, concise conclusion (not identical to summary)
    if headline:
        conclusion = headline
    elif recommendation:
        conclusion = recommendation
    else:
        conclusion = _first_sentence(summary_obj.get("summary", ""))

    return {
        "summary": summary_obj.get("summary", ""),
        "bullets": bullets,
        "conclusion": conclusion,
        "confidence": confidence,
        "citations": citations,
        "recommendation": recommendation,
        "headline": headline,
        "source_counts": source_counts,
    }


def run(profile_key: str, *, query: str, url: str) -> Dict[str, Any]:
    """High-level entry: load/create profile → gather → rank → synthesize → persist.

    Returns a structured bundle ready to be embedded into artifacts.
    """
    # create or load profile
    profile = store_mod.load_or_create(
        profile_key,
        seed_context={
            "query": query,
            "url": url,
        },
    )

    # execute engine: on-site evidence first
    evidence = gather(profile, query=query, url=url)

    # Optional external augmentation (opt-in via profile policies)
    pol = profile.get("source_policies", {})
    if bool(pol.get("augment", False)) and allow_network():
        try:
            from selfcoder.analysis.augment.external import gather_external
            allow = pol.get("external_allow") or []
            block = pol.get("external_block") or []
            max_external = int(pol.get("max_external", 6))
            timeout = int(pol.get("timeout", 10))
            _render = bool(pol.get("render", False))
            root_host = url
            try:
                # derive scheme://host from the root url
                from urllib.parse import urlparse
                pu = urlparse(url)
                root_host = f"{pu.scheme}://{pu.netloc}"
            except Exception:
                pass

            fresh_within = pol.get("fresh_within")
            ext_items = gather_external(
                query,
                root_host=root_host,
                allow=allow,
                block=block,
                max_pages=max_external,
                timeout=timeout,
                render=False,
                fresh_within=fresh_within,
            )
            if ext_items:
                # Merge, de-duplicate by normalized URL
                seen = { _norm_url(ev.get("url") or "") for ev in evidence if ev.get("url") }
                for ev in ext_items:
                    u = _norm_url(ev.get("url") or "")
                    if not u or u in seen:
                        continue
                    seen.add(u)
                    evidence.append(ev)
        except Exception:
            # If augmentation fails (missing extras or network), continue with on-site only
            pass

    ranked = rank(profile, query=query, evidence=evidence)
    result = synthesize(profile, query=query, ranked=ranked)

    # Consensus aggregation over top-ranked evidence (generic, transparent)
    try:
        evid_for_consensus = [
            {
                "summary": (ev.get("text") or "")[:400],
                "confidence": float(ev.get("score", 0.0)),
                "url": ev.get("url"),
                "name": _name_from_url(ev.get("url") or ""),
            }
            for ev in ranked[:3]
        ]
        consensus = aggregate_consensus(evid_for_consensus)
        result["consensus"] = consensus
        # Prefer consensus summary for headline/conclusion when present
        cons_text = (consensus.get("summary") or consensus.get("consensus") or "").strip()
        if cons_text:
            # Use a concise first sentence for headline/conclusion
            try:
                headline = _first_sentence(cons_text, max_len=180)
            except Exception:
                headline = cons_text[:180]
            if headline:
                result["headline"] = headline
                result["conclusion"] = headline
            # Small confidence nudge toward consensus (bounded)
            try:
                cons_conf = float(consensus.get("confidence") or 0.0)
                base_conf = float(result.get("confidence") or 0.0)
                if cons_conf and cons_conf > base_conf:
                    result["confidence"] = min(1.0, base_conf + 0.03)
            except Exception:
                pass
    except Exception:
        # Non-fatal; skip consensus if anything goes wrong
        pass

    # Surface consensus winner at the top level for easier downstream access
    try:
        cons = result.get("consensus", {}) or {}
        win = cons.get("winner", {}) or {}
        winner_name = win.get("name")
        winner_url = win.get("url")
        winner_source = win.get("source")
        winner_date_hint = win.get("date_hint")
    except Exception:
        winner_name = winner_url = winner_source = winner_date_hint = None

    # Prefer a concise winner-based headline/conclusion if consensus text is long or absent
    try:
        if winner_name:
            current_headline = result.get("headline", "") or ""
            # If no headline, too long, or starts with boilerplate (e.g., "what's …"), use compact winner line
            is_boiler = _norm_head(current_headline).startswith(_HEADLINE_BOILERPLATE_PREFIXES)
            if not current_headline or len(current_headline) > 160 or is_boiler:
                compact = f"For '{query}', {winner_name} is most relevant."
                result["headline"] = compact
                result["conclusion"] = compact
    except Exception:
        pass

    # 23e: diff vs last artifact and profile learning
    try:
        pol = profile.get("source_policies", {})
        topic = pol.get("persist_topic") or profile_key
        # Load last artifact for this topic from the knowledge index
        last = None
        try:
            idx = kb_index.load_index()
            cand = [e for e in idx if e.get("topic") == topic]
            cand.sort(key=lambda e: int(e.get("date", 0)), reverse=True)
            if cand:
                # skip the artifact we are about to write (not yet on disk), load the previous one if exists
                last_entry = cand[0]
                last_path = last_entry.get("artifact_path")
                if last_path:
                    import json as _json
                    from pathlib import Path as _Path
                    p = _Path(last_path)
                    if p.exists():
                        last = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            last = None

        if last and isinstance(last, dict):
            prev = last.get("result") or {}
            curr = result
            d = diff_artifacts(prev, curr)
            result["diff"] = d
            result["contradicted"] = bool(d.get("contradicted"))
            # Adjust profile weights and record calibration
            try:
                profile = adjust_weights(profile, d)
            except Exception:
                pass
            try:
                art_id = text_hash(result.get("conclusion") or result.get("headline") or "")
                track_calibration(profile, d, art_id)
            except Exception:
                pass
            # Persist updated profile weights/metrics
            try:
                store_mod.save(profile)
            except Exception:
                pass
    except Exception:
        pass

    # 23e: store weights snapshot for forensic diff
    try:
        result["weights_snapshot"] = {
            "weights": dict(profile.get("weights", {})),
            "scoring": dict(profile.get("scoring", {})),
        }
    except Exception:
        pass
    # Backward-compatible schema hint and top-level citations alias
    result["schema_version"] = 2
    top_level_citations = result.get("citations", [])

    # update profile metrics
    store_mod.touch(profile, last_confidence=result.get("confidence"))

    # Tier-3: expose a compact top-level diff snapshot (optional polish)
    try:
        _d = result.get("diff") or {}
        _diff_overlap = float(_d.get("overlap") or 0.0)
        _diff_contradicted = bool(_d.get("contradicted") or False)
    except Exception:
        _diff_overlap = 0.0
        _diff_contradicted = False

    return {
        "schema_version": 2,
        "winner_name": winner_name,
        "winner_url": winner_url,
        "winner_source": winner_source,
        "winner_date_hint": winner_date_hint,
        "profile_id": profile["profile_id"],
        "profile": profile,
        "evidence": evidence,
        "ranked": ranked,
        "result": result,
        "citations": top_level_citations,
        "diff_overlap": _diff_overlap,
        "diff_contradicted": _diff_contradicted,
    }
