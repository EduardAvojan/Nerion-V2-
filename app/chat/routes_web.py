from __future__ import annotations

from typing import Optional, Dict, Any
import re
from urllib.parse import urlparse as _urlparse

from app.chat.net_access import ensure_network_for as _ensure_net
from selfcoder.analysis.adapters import engine as _sq_engine
from selfcoder.analysis.search_api import search_enriched as _search_enriched
from app.chat.research import run_extraction
from selfcoder.analysis.knowledge import index as kb_index
from app.chat.intents import parse_site_query_intent
from app.chat.llm import (
    _answer_style_hint,
    _strip_think_blocks,
    build_chain_with_temp,
)
from app.chat.voice_io import safe_speak, listen_once
from app.chat.net_access import status_chip as _status_chip
from app.logging.experience import ExperienceLogger as _ExperienceLogger
from app.chat.ux_busy import BusyIndicator as _Busy
from selfcoder.config import allow_network as _allow_net
import os
import logging

_LOGGER = _ExperienceLogger()
DEBUG = bool((os.getenv('NERION_DEBUG') or '').strip())
_log = logging.getLogger(__name__)


def _ensure_network_for(task_type: str, watcher, url: Optional[str] = None) -> bool:
    """Shared ensure helper bridging to app.chat.net_access.ensure_network_for."""
    return _ensure_net(task_type, lambda m: safe_speak(m, watcher), listen_once, url=url, watcher=watcher)


def run_site_query(heard: str, STATE, watcher, parent_decision_dict: Optional[Dict[str, Any]]) -> bool:
    """Handle the site-query path. Returns True if the turn was handled."""
    intent = parse_site_query_intent(heard)
    if not intent:
        return False

    url = intent["url"]
    if not _ensure_network_for('site_query', watcher, url):
        return True  # handled (prompted)

    query = intent["query"]
    host = _urlparse(url).netloc
    profile_key = f"{host}:{query.lower()}"
    pol = {"render": True, "timeout": 15, "render_timeout": 7, "depth": 2, "max_pages": 10,
           "augment": bool(intent.get("augment")), "external_allow": intent.get("allow") or [],
           "fresh_within": "60d", "max_external": 4}
    prof = _sq_engine.store_mod.load_or_create(profile_key, seed_context={"query": query, "url": url, **pol})
    prof.setdefault("source_policies", {}).update(pol)
    _sq_engine.store_mod.save(prof)
    with _Busy("Reading that page…", start_delay_s=2.0):
        out = _sq_engine.run(profile_key, query=query, url=url)
    res = out.get("result", {}) or {}
    cons = res.get("consensus", {}) or {}
    win = cons.get("winner", {}) or {}
    winner_name = win.get("name")

    def _is_category_name(name: str) -> bool:
        nm = (name or "").strip().lower()
        return any(k in nm for k in ("next gen", "products", "solutions", "pcs", "pc"))

    def _is_productlike(name: str) -> bool:
        nm = (name or "").strip()
        return bool(re.search(r"\d", nm)) or bool(re.search(r"\b[a-zA-Z]\d|\d[a-zA-Z]\b", nm))

    spoken_winner = winner_name
    if _is_category_name(spoken_winner or ""):
        try:
            for c in (cons.get("candidates") or []):
                nm = (c or {}).get("name") or ""
                if nm and _is_productlike(nm):
                    spoken_winner = nm
                    break
        except Exception:
            pass

    spoken_q = re.sub(r"\s*(?:use|using|with|include)\s+(?:external\s+)?(?:reviews|sources)\s+from\s+.+$", "", query, flags=re.I).strip()
    head = (f"For '{spoken_q}', {spoken_winner} is most relevant." if spoken_winner else (res.get("headline") or res.get("conclusion") or "Done."))
    print('Nerion:', head)
    safe_speak(head, watcher)
    chip = _status_chip()
    print('Nerion:', chip)
    safe_speak(chip, watcher)

    conf = res.get("confidence")
    brief_bits = []
    rec_raw = res.get("recommendation") or ""
    rec = " ".join(re.sub(r"https?://\S+|\b[A-Za-z0-9-]+\.[A-Za-z]{2,}\b", "", re.sub(r"\s*Alternatives\s+highlighted\s+include\s+[^.]+\.?,?", "", rec_raw, flags=re.I))).strip(" ,.;")
    if rec:
        brief_bits.append(rec)
    alts = []
    try:
        cands = (cons.get("candidates") or [])
        for c in cands[1:5]:
            nm = (c or {}).get("name") or ""
            nm_clean = nm.strip()
            if not nm_clean:
                continue
            if re.search(r"https?://|\b[A-Za-z0-9-]+\.[A-Za-z]{2,}\b", nm_clean):
                continue
            if winner_name and nm_clean.lower() == str(winner_name).strip().lower():
                continue
            if any(nm_clean.lower() == a.lower() for a in alts):
                continue
            alts.append(nm_clean)
        alts = alts[:2]
    except Exception:
        pass
    if alts:
        brief_bits.append(f"Alternatives: {', '.join(alts)}.")

    if conf is not None:
        try:
            brief_bits.append(f"Confidence: {float(conf):.2f}.")
        except Exception:
            pass
    if brief_bits:
        brief = ' '.join(brief_bits)
        print('Nerion:', brief)
        safe_speak(brief, watcher)

    art = out.get("artifact_path") or res.get("artifact_path")
    if art:
        note = f"I saved the details to {art}"
        print('Nerion:', note)
        safe_speak(note, watcher)
        try:
            STATE.set_last_artifact_path(art)
        except Exception:
            pass

    # Notebook
    try:
        _LOGGER.log(
            user_query=heard,
            parent_decision=parent_decision_dict or {},
            action_taken={"routed": "site_query", "url": url},
            outcome_success=True,
            error=None,
            network_used=True,
        )
    except Exception:
        pass
    return True


def run_web_search(heard: str, STATE, watcher, parent_decision_dict: Optional[Dict[str, Any]]) -> bool:
    """Handle the open-web research path. Returns True if handled."""
    if not _allow_net():
        return False
    if not _ensure_network_for('web_search', watcher, None):
        return True

    try:
        low = heard.lower()
        freshness = None
        if 'today' in low or 'tomorrow' in low:
            freshness = 'day'
        elif 'week' in low:
            freshness = 'week'
        elif 'month' in low:
            freshness = 'month'
        with _Busy("Researching on the web…", start_delay_s=2.0):
            enriched = _search_enriched(heard, n=5, freshness=freshness, allow=None)
        urls = enriched.get('urls', [])
        structured = enriched.get('structured')
        if DEBUG:
            _log.debug("[Search] urls returned: %d", len(urls))
    except Exception:
        urls = []
        structured = None
    # Proceed even if search returns no URLs: allow test stubs and cached data
    if urls:
        with _Busy("Extracting key information…", start_delay_s=2.0):
            artifacts, snippet_block = run_extraction(urls, heard, structured)
    else:
        try:
            # Fallback: attempt extraction with an empty URL list (enables unit-test stubs
            # and cached summaries to exercise downstream formatting and notes.)
            artifacts, snippet_block = run_extraction([], heard, structured)
        except Exception:
            artifacts, snippet_block = [], ""
        try:
            temp = STATE.voice.current_temperature(0.22)
            chat_chain = build_chain_with_temp(temp)
            style = _answer_style_hint(heard)
            prompt = (
                "You are Nerion. Using ONLY the CONTEXT below, write a single, direct, helpful answer to the user's question.\n"
                "Be concrete. Quote numbers/dates exactly as shown. If sources disagree, prefer the most recent or majority view.\n"
                "Cite at most 1–2 domains by name in parentheses at the end (e.g., (siteA; siteB)).\n"
                "Do NOT mention internal IDs, 'source #' labels, or the word 'artifact'. Do NOT invent facts.\n\n"
                "CONTEXT:\n"
                f"{snippet_block}\n\n"
                "TASK:\n"
                f"{style} Answer the user’s question directly in 1–2 sentences.\n\n"
                f"Question: {heard}\n\n"
                "Answer:"
            )
            with _Busy("Writing up what I found…", start_delay_s=2.0):
                response = chat_chain.predict(input=prompt)
        except Exception as e:
            response = f"I tried researching that, but hit an issue: {e}"
        response = _strip_think_blocks(response).strip()
        print('Nerion:', response)
        safe_speak(response, watcher)
        chip = _status_chip()
        print('Nerion:', chip)
        safe_speak(chip, watcher)
        # Staleness note (cache age) — if any sources came from cache, surface a brief note
        try:
            ages = []
            for a in artifacts or []:
                try:
                    ages.append(int(a.get('cache_age_s') or 0))
                except Exception:
                    continue
            if ages:
                max_age = max(ages)
                if max_age > 0:
                    # Humanize up to days
                    def _fmt_age(secs: int) -> str:
                        try:
                            if secs < 3600:
                                m = max(1, secs // 60)
                                return f"~{m}m"
                            if secs < 86400:
                                h = max(1, secs // 3600)
                                return f"~{h}h"
                            d = max(1, secs // 86400)
                            return f"~{d}d"
                        except Exception:
                            return ""
                    note = f"Note: some sources are from cache ({_fmt_age(max_age)} old)."
                    print('Nerion:', note)
                    safe_speak(note, watcher)
        except Exception:
            pass
        try:
            STATE.append_turn('assistant', response)
        except Exception:
            pass
        # RAG-ready: persist a lightweight index entry and per-URL chunks
        try:
            topic = f"search:{heard.strip()[:60]}"
            kb_index.append_entry({
                "topic": topic,
                "domain": "web_search",
                "query": heard,
                "artifact_path": None,
                "confidence": None,
            })
            for a in artifacts[:5]:
                extract = (a.get('snippet') or a.get('headline') or a.get('recommendation') or a.get('winner') or '')
                if not extract:
                    continue
                try:
                    kb_index.append_chunk({
                        "topic": topic,
                        "domain": "web_search",
                        "url": a.get('url'),
                        "extract": extract,
                    })
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _LOGGER.log(
                user_query=heard,
                parent_decision=parent_decision_dict or {},
                action_taken={"routed": "web_search", "urls": urls[:3]},
                outcome_success=True,
                error=None,
                network_used=True,
            )
        except Exception:
            pass
        return True
    return False
