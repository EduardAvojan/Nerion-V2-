import sys

import pytest

from ops.security.net_gate import NetworkGate
from selfcoder.analysis import web_render as WR


class DummyResp:
    def __init__(self, text: str):
        self.text = text
    def raise_for_status(self):
        return None


@pytest.fixture(autouse=True)
def reset_gate():
    NetworkGate.revoke(reason="test setup")
    NetworkGate.init({"allow_network_access": True, "net": {"idle_revoke_after": "15m"}})
    yield
    NetworkGate.revoke(reason="test teardown")


def test_render_url_offline_blocks(monkeypatch):
    """Gate should prevent any HTTP attempt when offline."""
    calls = {"httpx": 0, "urllib": 0}

    def fake_httpx_get(*args, **kwargs):
        calls["httpx"] += 1
        raise AssertionError("httpx.get should not be called when gate is offline")

    class _DummyURL:
        def __init__(self, *a, **k):
            calls["urllib"] += 1
            raise AssertionError("urllib.request.urlopen should not be called when gate is offline")

    # Patch both httpx and urllib fallbacks
    monkeypatch.setattr(WR, "httpx", type("H", (), {"get": staticmethod(fake_httpx_get)}))
    import types as _t
    def _urlopen(*a, **k):
        calls["urllib"] += 1
        raise AssertionError("urlopen should not be called when gate is offline")
    UR = _t.SimpleNamespace(Request=lambda url, headers=None: (url, headers), urlopen=_urlopen)
    monkeypatch.setattr(WR, "urlopen", UR.urlopen, raising=False)
    monkeypatch.setattr(WR, "Request", UR.Request, raising=False)

    with pytest.raises(PermissionError):
        WR.render_url("https://example.com", render_timeout=1, http_timeout=1)

    assert calls["httpx"] == 0
    assert calls["urllib"] == 0


def test_render_url_allowed_fetches(monkeypatch):
    """With a session grant, render_url should fetch and return HTML text via httpx path."""
    NetworkGate.grant_session(minutes=10, reason="render ok")

    def fake_httpx_get(url, headers=None, timeout=5, follow_redirects=True):
        return DummyResp("<html><body>OK</body></html>")

    monkeypatch.setattr(WR, "httpx", type("H", (), {"get": staticmethod(fake_httpx_get)}))

    # Force the test to use the httpx fallback by making playwright fail to import
    monkeypatch.setitem(sys.modules, "playwright", None)

    out = WR.render_url("https://example.com/page", render_timeout=1, http_timeout=1)
    assert isinstance(out, str)
    assert "OK" in out