from __future__ import annotations

import os
import types

from app.chat import voice_io as V


class _DummyMic:
    def __init__(self, *a, **k):
        pass
    def __enter__(self):
        return object()
    def __exit__(self, exc_type, exc, tb):
        return False


def test_strict_offline_returns_none_when_sphinx_fails(monkeypatch):
    # Force offline + strict offline
    monkeypatch.setenv("NERION_STT_OFFLINE", "1")
    monkeypatch.setenv("NERION_STT_STRICT_OFFLINE", "1")

    # Replace Microphone with a dummy context
    monkeypatch.setattr(V, "sr", types.SimpleNamespace(Microphone=_DummyMic), raising=False)

    # Stub recognizer.listen to return a dummy audio object
    monkeypatch.setattr(V._recognizer, "listen", lambda *a, **k: object(), raising=False)

    # Make recognize_sphinx raise and recognize_google if called would raise (to detect fallback)
    monkeypatch.setattr(V._recognizer, "recognize_sphinx", lambda *a, **k: (_ for _ in ()).throw(Exception("no sphinx")), raising=False)

    called_google = {"n": 0}
    def _google(*a, **k):
        called_google["n"] += 1
        raise RuntimeError("should not be called in strict offline")
    monkeypatch.setattr(V._recognizer, "recognize_google", _google, raising=False)

    out = V.listen_once(timeout=0.1, phrase_time_limit=0.1)
    assert out is None
    assert called_google["n"] == 0
