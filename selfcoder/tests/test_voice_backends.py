from __future__ import annotations

import types
from pathlib import Path


def test_tts_resolve_backend_env_overrides(monkeypatch):
    import app.chat.tts_router as tts
    # env wins over preference
    monkeypatch.setenv('NERION_TTS_BACKEND', 'piper')
    assert tts.resolve_backend('say') == 'piper'
    monkeypatch.delenv('NERION_TTS_BACKEND', raising=False)


def test_tts_resolve_backend_pref_and_platform(monkeypatch):
    import app.chat.tts_router as tts
    # explicit preference honored
    assert tts.resolve_backend('say') == 'say'
    assert tts.resolve_backend('pyttsx3') == 'pyttsx3'
    # auto platform default: darwin -> say, else pyttsx3
    monkeypatch.setenv('NERION_FORCE_SAY', '')
    monkeypatch.delenv('NERION_TTS_BACKEND', raising=False)
    # Pretend macOS
    monkeypatch.setattr(tts, 'sys', types.SimpleNamespace(platform='darwin'))
    assert tts.resolve_backend(None) == 'say'
    # Pretend Linux
    monkeypatch.setattr(tts, 'sys', types.SimpleNamespace(platform='linux'))
    assert tts.resolve_backend(None) == 'pyttsx3'


def test_tts_speak_routes(monkeypatch):
    import app.chat.tts_router as tts
    # Ensure clean state
    tts.reset()
    called = {'say': 0, 'piper': 0, 'coqui': 0}
    monkeypatch.setattr(tts, '_speak_via_say', lambda text: called.__setitem__('say', called['say'] + 1))
    monkeypatch.setattr(tts, '_speak_via_piper', lambda text: called.__setitem__('piper', called['piper'] + 1))
    monkeypatch.setattr(tts, '_speak_via_coqui', lambda text: called.__setitem__('coqui', called['coqui'] + 1))

    # Route to say
    tts._backend_choice = 'say'
    tts.speak('hello')
    assert called['say'] == 1

    # Route to piper
    tts._backend_choice = 'piper'
    tts.speak('hello')
    assert called['piper'] == 1

    # Route to coqui
    tts._backend_choice = 'coqui'
    tts.speak('hello')
    assert called['coqui'] == 1

    # pyttsx3 path with no engine -> fallback to say
    tts._backend_choice = 'pyttsx3'
    monkeypatch.setattr(tts, '_engine', None)
    tts.speak('hello')
    assert called['say'] == 2

    # pyttsx3 path with engine -> enqueues text (no external call)
    class DummyEngine:
        def stop(self):
            pass
    tts._engine = DummyEngine()
    # drain queue if anything present
    try:
        while True:
            tts._tts_queue.get_nowait()
    except Exception:
        pass
    tts.speak('enqueued')
    # Should not increase any backend counters
    assert called['say'] == 2 and called['piper'] == 1 and called['coqui'] == 1
    # The queue should now have the text
    got = tts._tts_queue.get(timeout=1)
    assert got == 'enqueued'


def _frames_gen(chunks=3, size=320):
    for _ in range(chunks):
        yield b'\x00' * size


def test_stt_backend_dispatch(monkeypatch):
    import voice.stt.recognizer as rec
    # Monkeypatch backends to return sentinel values
    monkeypatch.setattr(rec, '_try_whisper', lambda *a, **k: 'W')
    monkeypatch.setattr(rec, '_try_whispercpp', lambda *a, **k: 'C')
    monkeypatch.setattr(rec, '_try_vosk', lambda *a, **k: 'V')
    monkeypatch.setattr(rec, '_try_sphinx', lambda *a, **k: 'S')

    # Explicit backend via set_profile
    rec.set_profile('whisper', 'small')
    out = rec.transcribe_streaming(_frames_gen(), sample_rate=16000)
    assert out == 'W'

    rec.set_profile('whisper.cpp', None)
    out = rec.transcribe_streaming(_frames_gen(), sample_rate=16000)
    assert out == 'C'

    rec.set_profile('vosk', 'small')
    out = rec.transcribe_streaming(_frames_gen(), sample_rate=16000)
    assert out == 'V'

    # Fallback to sphinx when backend unknown
    rec.set_profile('auto', None)
    monkeypatch.setenv('NERION_STT_BACKEND', 'sphinx')
    out = rec.transcribe_streaming(_frames_gen(), sample_rate=16000)
    assert out == 'S'

