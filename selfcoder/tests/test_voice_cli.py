from __future__ import annotations

import sys
from pathlib import Path


def test_voice_devices_command_runs(monkeypatch, capsys):
    from selfcoder.cli_ext import voice as v
    # Run devices regardless of PyAudio presence; should not raise
    rc = 0
    try:
        rc = v.register  # ensure module import ok
        # Call the inner function via parser
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest='cmd')
        v.register(sub)
        # Simulate: nerion voice devices
        ns = parser.parse_args(['voice', 'devices'])
        rc = ns.func(ns)
    except SystemExit:
        rc = 0
    assert rc == 0
    out = capsys.readouterr().out
    # It should print either a header or a helpful unavailability message
    assert ('[voice] input devices' in out) or ('device enumeration unavailable' in out)


def test_env_validation_warnings_piper(monkeypatch, capsys):
    import app.chat.tts_router as tts
    tts.reset()
    # Force backend piper with missing model & CLI
    monkeypatch.delenv('PIPER_MODEL_PATH', raising=False)
    monkeypatch.setenv('NERION_TTS_BACKEND', 'piper')
    monkeypatch.setattr(tts, '_which', lambda _: None)
    tts.init_tts(None, rate=180, preferred_voice='Daniel')
    out = capsys.readouterr().out
    assert 'PIPER_MODEL_PATH' in out or 'piper CLI not found' in out


def test_env_validation_warnings_whispercpp(monkeypatch, capsys):
    import voice.stt.recognizer as rec
    monkeypatch.delenv('WHISPER_CPP_MODEL_PATH', raising=False)
    monkeypatch.delenv('NERION_WHISPER_CPP_MODEL', raising=False)
    # Reset one-time warning guard to ensure print occurs in this test
    try:
        rec._WARNED_MISSING['whispercpp'] = False
    except Exception:
        pass
    _ = capsys.readouterr()  # clear any prior output
    rec.set_profile('whisper.cpp', None)
    out = capsys.readouterr().out
    assert 'WHISPER_CPP_MODEL_PATH' in out
