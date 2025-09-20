

import types
import pytest

from ops.security.net_gate import NetworkGate
from selfcoder.analysis import search_api as SA


class DummyResp:
    def __init__(self, json_obj):
        self._json = json_obj

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def reset_gate(monkeypatch):
    # Ensure clean gate each test
    NetworkGate.revoke(reason="test setup")
    NetworkGate.init({"allow_network_access": True, "net": {"idle_revoke_after": "15m"}})
    yield
    NetworkGate.revoke(reason="test teardown")


def test_duck_offline_blocks(monkeypatch):
    """When offline (no grant), provider should be blocked before any HTTP occurs."""
    called = {"get": 0}

    def fake_get(*args, **kwargs):
        called["get"] += 1
        raise AssertionError("requests.get should not be reached when gate is offline")

    monkeypatch.setattr(SA.requests, "get", fake_get)

    with pytest.raises(PermissionError):
        SA._search_duck("time now", n=3, freshness=None, allow=None)

    assert called["get"] == 0  # assert gate tripped before network


def test_duck_allowed_returns_urls(monkeypatch):
    """With a session grant, the call should proceed and parse minimal JSON into URLs."""
    NetworkGate.grant_session(minutes=10, reason="test duck ok")

    def fake_get(*args, **kwargs):
        # Minimal shape that _search_duck expects: either RelatedTopics or AbstractURL
        return DummyResp({
            "RelatedTopics": [
                {"FirstURL": "https://example.com/a"},
                {"FirstURL": "https://example.com/b"},
            ]
        })

    monkeypatch.setattr(SA.requests, "get", fake_get)

    out = SA._search_duck("something", n=2, freshness=None, allow=None)
    assert isinstance(out, list)
    assert out[:2] == ["https://example.com/a", "https://example.com/b"]


def test_domain_scope_blocks_duck(monkeypatch):
    """Grant restricted to example.com should still block duck API domain (api.duckduckgo.com)."""
    NetworkGate.grant_session(minutes=10, domains=["example.com"], reason="scoped")

    called = {"get": 0}

    def fake_get(*args, **kwargs):
        called["get"] += 1
        return DummyResp({"RelatedTopics": []})

    monkeypatch.setattr(SA.requests, "get", fake_get)

    with pytest.raises(PermissionError):
        SA._search_duck("anything", n=1, freshness=None, allow=None)

    assert called["get"] == 0