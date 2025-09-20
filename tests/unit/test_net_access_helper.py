from __future__ import annotations

import os
from typing import Optional

import types

from app.chat import net_access as NA


def _speak_capture(buf):
    def _f(msg: str):
        buf.append(str(msg))
    return _f


def test_ensure_network_for_yes_grants(monkeypatch, tmp_path):
    # Isolate prefs path
    monkeypatch.setattr(NA, "_PREFS_PATH", str(tmp_path / "prefs.json"), raising=False)

    # Gate not allowed initially
    monkeypatch.setattr(NA.NetworkGate, "can_use", lambda *a, **k: False)

    called = {"granted": False}

    def _grant(**kwargs):
        called["granted"] = True
    monkeypatch.setattr(NA.NetworkGate, "grant_session", lambda **kw: _grant(**kw))

    out = []
    ok = NA.ensure_network_for(
        "web_search",
        speak=_speak_capture(out),
        listen_once=lambda **kw: "yes",
    )
    assert ok is True
    assert called["granted"] is True


def test_ensure_network_for_always_persists(monkeypatch, tmp_path):
    # Isolate prefs path
    monkeypatch.setattr(NA, "_PREFS_PATH", str(tmp_path / "prefs.json"), raising=False)

    # Start not granted
    monkeypatch.setattr(NA.NetworkGate, "can_use", lambda *a, **k: False)
    monkeypatch.setattr(NA.NetworkGate, "grant_session", lambda **kw: None)

    out = []
    task = "site_query"
    ok = NA.ensure_network_for(
        task,
        speak=_speak_capture(out),
        listen_once=lambda **kw: None,
        typed_fallback=True,
    )
    # No voice reply and no typed input leads to staying offline
    assert ok is False

    # Now simulate typed input path
    inputs = iter(["always"])  # one-shot "always"
    import builtins
    monkeypatch.setattr(builtins, "input", lambda *_a, **_k: next(inputs))
    ok2 = NA.ensure_network_for(
        task,
        speak=_speak_capture(out),
        listen_once=lambda **kw: None,
        typed_fallback=True,
    )
    assert ok2 is True
    prefs = NA.load_net_prefs()
    assert bool((prefs.get("always_allow_by_task") or {}).get(task)) is True
