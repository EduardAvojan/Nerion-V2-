from __future__ import annotations

import os
from selfcoder.analysis.search_api import normalize_search_env


def test_normalize_search_env_maps_serpapi(monkeypatch):
    monkeypatch.delenv("NERION_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("NERION_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "abc123")

    prov, has_key = normalize_search_env()
    assert has_key is True
    assert prov == "serpapi"
    assert os.getenv("NERION_SEARCH_API_KEY") == "abc123"


def test_normalize_search_env_defaults_duck(monkeypatch):
    for k in ("SERPAPI_API_KEY", "NERION_SEARCH_API_KEY", "NERION_SEARCH_PROVIDER"):
        monkeypatch.delenv(k, raising=False)

    prov, has_key = normalize_search_env()
    assert has_key is False
    assert prov in {"duck", ""}  # provider may be empty if env set later

