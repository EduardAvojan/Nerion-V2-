from __future__ import annotations

from typing import List, Dict, Any
import types

import app.chat.research as R


def _stub_run(profile_key: str, query: str, url: str) -> Dict[str, Any]:
    # Craft different dates in headline/snippet and differing numeric claims
    if "site-new.com" in url:
        return {
            "result": {
                "headline": "Updated Aug 20, 2025",
                "recommendation": "",
                "confidence": 0.9,
                "consensus": {"winner": {"name": "Prod 123"}},
            },
            "citations": [{"source": "external", "url": "https://site-new.com/ref"}],
        }
    if "site-old.com" in url:
        return {
            "result": {
                "headline": "Last revised 2021-01-01",
                "recommendation": "",
                "confidence": 0.5,
                "consensus": {"winner": {"name": "Prod 123"}},
            },
            "citations": [{"source": "external", "url": "https://site-old.com/ref"}],
        }
    # Fallback minimal
    return {"result": {"headline": "", "recommendation": ""}}


def _stub_fetch(url: str, query: str, max_chars: int = 1200) -> str:
    if "site-new.com" in url:
        return "Battery life is 12 hours."
    if "site-old.com" in url:
        return "Battery life is 10 hours."
    return ""


def test_run_extraction_sorts_by_recency_and_skips_paywalls(monkeypatch):
    # Monkeypatch the site-query engine and snippet fetcher
    monkeypatch.setattr(R._sq_engine, "run", _stub_run, raising=False)
    monkeypatch.setattr(R, "_fetch_snippet", _stub_fetch, raising=False)

    urls = [
        "https://www.wsj.com/paywalled-article",  # should be skipped
        "https://m.site-new.com/page1",
        "https://site-old.com/page2",
        "https://amp.site-new.com/page-dup",  # duplicate host; should be deduped
    ]
    arts, block = R.run_extraction(urls, query="test product")

    # First artifact should be the newer site
    assert any("site-new.com" in (a.get("url") or "") for a in arts[:1])
    # No WSJ artifacts
    assert all("wsj.com" not in (a.get("url") or "") for a in arts)
    # Contradiction note appears (12h vs 10h)
    assert ("Contradiction flagged" in block) or ("contradiction flagged" in block.lower())

