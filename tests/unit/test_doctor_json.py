from __future__ import annotations

from typing import Dict, Any


def test_doctor_json_structure(monkeypatch):
    from selfcoder.cli_ext import doctor as doc

    # Stub out environment-dependent checks
    monkeypatch.setattr(doc, "_check_import", lambda name: (True, f"import {name}"), raising=False)

    monkeypatch.setattr(
        doc,
        "_check_microphones",
        lambda: (True, {"component": "microphone", "ok": True, "detail": "1 device(s) detected", "devices": ["Test Mic"], "remedy": []}),
        raising=False,
    )

    monkeypatch.setattr(
        doc,
        "_check_tts",
        lambda: (True, {"component": "tts", "ok": True, "detail": "pyttsx3: OK; say: OK", "remedy": []}),
        raising=False,
    )

    monkeypatch.setattr(
        doc,
        "_check_offline_stt",
        lambda: (True, {"component": "offline_stt", "ok": True, "detail": "pocketsphinx present", "remedy": []}),
        raising=False,
    )

    rep: Dict[str, Any] = doc._doctor_report()
    assert isinstance(rep, dict)
    assert "ok" in rep and isinstance(rep["ok"], bool)
    assert "items" in rep and isinstance(rep["items"], list)

    comps = {it.get("component") for it in rep["items"]}
    assert {"microphone", "tts", "offline_stt"}.issubset(comps)
    assert rep["ok"] is True

