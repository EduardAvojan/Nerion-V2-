from __future__ import annotations

import builtins
from types import SimpleNamespace

import app.chat.routes_web as rw


class _StubChain:
    def predict(self, input: str) -> str:
        return "Answer"


class _Voice:
    def current_temperature(self, _d):
        return 0.2


class _State:
    voice = _Voice()
    def append_turn(self, *a, **k):
        pass


def test_staleness_note_is_printed(monkeypatch):
    # Network always allowed and ensure_network returns True
    monkeypatch.setattr(rw, "_allow_net", lambda: True, raising=False)
    monkeypatch.setattr(rw, "_ensure_network_for", lambda *a, **k: True, raising=False)
    # Stub build_chain_with_temp
    monkeypatch.setattr(rw, "build_chain_with_temp", lambda _t: _StubChain(), raising=False)
    # Stub run_extraction to return artifacts with cache_age_s
    artifacts = [
        {'url': 'https://x', 'headline': '', 'recommendation': '', 'winner': '', 'confidence': 0.5, 'domains': [], 'snippet': 'Battery 10h', 'cache_age_s': 7200},
    ]
    monkeypatch.setattr(rw, "run_extraction", lambda urls, heard, structured=None: (artifacts, 'Battery 10h'), raising=False)

    # Capture prints
    printed = []
    monkeypatch.setattr(builtins, 'print', lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    ok = rw.run_web_search("test query", _State(), None, None)
    assert ok is True
    assert any("sources are from cache" in ln for ln in printed)

