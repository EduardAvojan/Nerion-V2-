from __future__ import annotations

from typing import List

from app.chat.commands import try_parse_command, handle_command


def _capture_speak(out: List[str]):
    def _f(msg: str):
        out.append(str(msg))
    return _f


def test_try_parse_stt_command_parses_backend_and_model():
    c, a = try_parse_command("/stt whisper small")
    assert c == "stt"
    assert a.strip().lower() == "whisper small"


def test_handle_stt_sets_profile_and_acknowledges():
    calls = {"backend": None, "model": None}
    out: List[str] = []

    def _set_stt(b, m):
        calls["backend"] = b
        calls["model"] = m

    ok = handle_command(
        "stt",
        "whisper small",
        speak_fn=_capture_speak(out),
        set_speech_enabled=lambda v: None,
        get_speech_enabled=lambda: True,
        set_mute_fn=lambda m: None,
        set_tts_params=lambda r, v: None,
        set_stt_profile_fn=_set_stt,
    )

    assert ok is True
    assert calls["backend"] == "whisper"
    assert calls["model"] == "small"
    assert any("stt set to whisper small" in s.lower() for s in out)


def test_handle_stt_unknown_backend_emits_help():
    out: List[str] = []
    ok = handle_command(
        "stt",
        "unknown",
        speak_fn=_capture_speak(out),
        set_speech_enabled=lambda v: None,
        get_speech_enabled=lambda: True,
        set_mute_fn=lambda m: None,
        set_tts_params=lambda r, v: None,
        set_stt_profile_fn=lambda b, m: None,
    )
    assert ok is True
    assert any("unknown stt backend" in s.lower() for s in out)

