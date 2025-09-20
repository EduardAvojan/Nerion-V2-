from __future__ import annotations

from selfcoder import cli


def test_voice_diagnose_cli_success(monkeypatch):
    import voice.stt.recognizer as rec
    monkeypatch.setattr(rec, "run_diagnostics", lambda **kw: True, raising=False)
    rc = cli.main(["voice", "diagnose", "--duration", "1"]) 
    assert rc == 0


def test_voice_diagnose_cli_failure(monkeypatch):
    import voice.stt.recognizer as rec
    monkeypatch.setattr(rec, "run_diagnostics", lambda **kw: False, raising=False)
    rc = cli.main(["voice", "diagnose", "--duration", "1"]) 
    assert rc == 1

