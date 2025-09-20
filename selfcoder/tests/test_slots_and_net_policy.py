from __future__ import annotations

import re
from typing import Optional

import pytest

from app.chat.slots import parse_task_slots
from app.chat import net_policy as netp
from ops.security.net_gate import NetworkGate
import time


# ---------------------- parse_task_slots tests ----------------------

def test_parse_task_slots_best_pick():
    out = parse_task_slots("What's the best laptop this month?")
    assert out is not None
    assert out.get("intent") == "best_pick"
    assert out.get("timeframe") == "this month"
    assert "laptop" in (out.get("item") or "").lower()
    assert 0.0 < float(out.get("confidence") or 0) <= 1.0


def test_parse_task_slots_fact_lookup_location():
    out = parse_task_slots("what's the current weather today in Boston")
    assert out is not None
    assert out.get("intent") == "fact_lookup"
    assert out.get("timeframe") == "today"
    # Location extraction is heuristic; verify we picked up a token
    assert (out.get("location") or "").lower() == "boston"


def test_parse_task_slots_compare_and_sources():
    url = "https://example.com/article"
    out = parse_task_slots(f"compare iphone vs pixel and summarize {url}")
    assert out is not None
    # either compare or summarize depending on first keyword; accept either
    assert out.get("intent") in {"compare", "summarize"}
    item = (out.get("item") or "").lower()
    assert "iphone" in item and "pixel" in item
    assert url in (out.get("sources") or [])


def test_parse_task_slots_no_match():
    assert parse_task_slots("") is None
    assert parse_task_slots("hello there") is None


# ---------------------- net_policy tests ----------------------------

def test_net_policy_denied_when_master_off(monkeypatch):
    # Force master switch off regardless of NetworkGate
    monkeypatch.setattr(netp, "allow_network", lambda: False, raising=False)
    NetworkGate.revoke()
    assert netp.can_use("web.search") is False


def test_net_policy_request_session_and_can_use(monkeypatch):
    # Enable master switch
    monkeypatch.setattr(netp, "allow_network", lambda: True, raising=False)
    # Reset NetworkGate state
    NetworkGate.revoke()
    # Initially cannot use without a grant
    assert netp.can_use("web.search") is False
    # Grant a session and verify allow
    netp.request_session("web.search", minutes=1, reason="test")
    assert netp.can_use("web.search") is True
    # Revoke and ensure it's off again
    NetworkGate.revoke()
    assert netp.can_use("web.search") is False


def test_net_policy_domain_scoped_session(monkeypatch):
    # Enable master switch
    monkeypatch.setattr(netp, "allow_network", lambda: True, raising=False)
    NetworkGate.revoke()
    # Grant scoped to example.com only
    netp.request_session("web.search", minutes=1, domains=["example.com"], reason="scoped")
    assert netp.can_use("web.search", url="https://example.com/page") is True
    assert netp.can_use("web.search", url="https://other.com/page") is False
    NetworkGate.revoke()


def test_net_policy_domain_scoped_idle_timeout(monkeypatch):
    monkeypatch.setattr(netp, "allow_network", lambda: True, raising=False)
    # Configure idle revoke to a very short window
    NetworkGate.init({"net": {"idle_revoke_after": "1s"}})
    NetworkGate.revoke()
    netp.request_session("web.search", minutes=1, domains=["example.com"], reason="idle-test")
    # Initially allowed
    assert netp.can_use("web.search", url="https://example.com/page") is True
    # Wait past idle threshold; next check should revoke
    time.sleep(1.1)
    assert netp.can_use("web.search", url="https://example.com/page") is False
    NetworkGate.revoke()
