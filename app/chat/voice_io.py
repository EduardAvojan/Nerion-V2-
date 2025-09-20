

"""Voice I/O utilities for Nerion.
Decouples speaking, PTT-safe speak scheduling, mic capture, and calibration.
This module is topic-agnostic and avoids importing the runner to prevent cycles.
The runner must call `set_voice_state(STATE)` and `set_device_index(idx)` during init.
"""
from __future__ import annotations
import os
import re
import time
import threading
from typing import Optional

# Optional dependency: SpeechRecognition. Make import optional so CI without
# audio/voice deps can still import this module for helpers like _normalize_vocab.
try:  # pragma: no cover - import variability depends on environment
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover
    sr = None  # type: ignore

# TTS router imports kept local to avoid cycles with the runner
from .tts_router import speak as tts_speak, cancel_speech as tts_cancel, set_params as tts_set_params

# Optional: pynput presence for PTT awareness
try:
    from pynput import keyboard as _kb  # noqa: F401
    _HAS_PYNPUT = True
except Exception:  # pragma: no cover
    _HAS_PYNPUT = False

# Streaming STT (offline with VAD) and profile control
try:
    from voice.stt.recognizer import transcribe_streaming  # type: ignore
    from voice.stt.recognizer import set_profile as _stt_set_profile  # type: ignore
except Exception:  # pragma: no cover
    transcribe_streaming = None  # type: ignore
    def _stt_set_profile(*_a, **_k):  # type: ignore
        return None

# Injected at runtime by the runner
_STATE = None  # ChatState instance
_sr_device_index: Optional[int] = None


def _emit_chat_turn(text: str) -> None:
    """Mirror assistant text to the Electron UI even if TTS is skipped."""
    if not text:
        return
    try:
        from . import ipc_electron as _ipc
        if _ipc.enabled():
            _ipc.emit('chat_turn', {'role': 'assistant', 'text': str(text)})
    except Exception:
        pass


def set_voice_state(state_obj) -> None:
    """Inject the central ChatState so this module can read speech/mute/PTT flags."""
    global _STATE
    _STATE = state_obj


def set_device_index(idx: Optional[int]) -> None:
    """Inject the selected microphone index for speech_recognition."""
    global _sr_device_index
    _sr_device_index = idx


# -------------------------- Speak / Cancel / Safe Speak ----------------------

def _voice_status_enabled() -> bool:
    try:
        return (os.getenv('NERION_VOICE_STATUS') or '0').strip().lower() in {'1','true','yes','on'}
    except Exception:
        return False


def speak(text: str, *, force: bool = False) -> None:
    """Speak via the TTS router, honoring ChatState voice/mute unless force=True."""
    if not text:
        return
    try:
        enabled = bool(getattr(getattr(_STATE, 'voice', None), 'enabled', True))
        muted = bool(getattr(_STATE, 'muted', False))
    except Exception:
        enabled = True
        muted = False
    if (not force) and ((not enabled) or muted):
        _emit_chat_turn(text)
        return
    try:
        tts_speak(text)
        _notify_ui('audio_state', {'status': 'speaking'})
    except Exception as e:  # graceful fallback on AUHAL/PaMacCore hiccups
        msg = str(e) or ''
        if ('AUHAL' in msg) or ('PaMacCore' in msg) or ('-50' in msg):
            try:
                time.sleep(0.35)
                tts_speak(text)
                _notify_ui('audio_state', {'status': 'speaking'})
                _emit_chat_turn(text)
                return
            except Exception:
                if _voice_status_enabled():
                    print('[TTS] audio device hiccup — showing text only this turn')
                _emit_chat_turn(text)
                return
        if _voice_status_enabled():
            print('[TTS] speak unavailable — proceeding without audio')
        _emit_chat_turn(text)
        return
    _emit_chat_turn(text)


def cancel_speech() -> None:
    try:
        tts_cancel()
    except Exception:
        pass
    _notify_ui('audio_state', {'status': 'idle'})


def _stable_space_up(watcher, stable_ms: int = 120) -> bool:
    """Return True if SPACE has been continuously up for stable_ms milliseconds."""
    if watcher is None or not getattr(watcher, 'space_down', None):
        return True
    if watcher.space_down.is_set():
        return False
    t0 = time.time()
    while (time.time() - t0) < (stable_ms / 1000.0):
        if watcher.space_down.is_set():
            return False
        time.sleep(0.01)
    return True


def _schedule_speak_after_release(text: str, watcher, stable_ms: int = 150) -> None:
    """Background helper: wait until SPACE is stably released, then speak once."""
    def _runner():
        while watcher and getattr(watcher, 'space_down', None) and watcher.space_down.is_set():
            time.sleep(0.02)
        if not _stable_space_up(watcher, stable_ms=stable_ms):
            return
        if watcher and getattr(watcher, 'space_down', None) and watcher.space_down.is_set():
            return
        speak(text)
    th = threading.Thread(target=_runner, name='safe_speak_deferred', daemon=True)
    th.start()


def _adaptive_rate_for(text: str, base_rate: int) -> int:
    try:
        n = len(str(text or "").strip())
        if n <= 60:
            return int(base_rate * 1.08)
        if n >= 280:
            return int(base_rate * 0.92)
        return base_rate
    except Exception:
        return base_rate


def safe_speak(text: str, watcher=None, delay_s: float = 0.9, *, force: bool = False) -> None:
    """Speak unless the user is actively holding PTT (SPACE).

    - If SPACE is down, wait up to `delay_s` for release, then require a brief stable
      window to avoid key-repeat races. If still held, schedule a deferred speak.
    - If `force=True`, bypass PTT checks.
    """
    if not text:
        return
    try:
        enabled = bool(getattr(getattr(_STATE, 'voice', None), 'enabled', True))
    except Exception:
        enabled = True
    if (not force) and (not enabled):
        _emit_chat_turn(text)
        return
    if force:
        speak(text, force=True)
        return
    try:
        is_ptt = bool(getattr(getattr(_STATE, 'voice', None), 'ptt', True))
    except Exception:
        is_ptt = True
    # Adaptive rate: nudge TTS rate based on utterance length
    # We apply before speaking and do not attempt to restore here, since set_params is cheap and
    # future utterances will recompute.
    try:
        from app.nerion_chat import TTS_RATE as _BASE
        adaptive_rate = _adaptive_rate_for(text, int(_BASE))
        tts_set_params(rate=adaptive_rate)
    except Exception:
        pass
    try:
        if is_ptt and _HAS_PYNPUT and watcher is not None and getattr(watcher, 'space_down', None):
            t0 = time.time()
            while watcher.space_down.is_set() and (time.time() - t0) < delay_s:
                time.sleep(0.02)
            if not watcher.space_down.is_set():
                if not _stable_space_up(watcher, stable_ms=140):
                    if _voice_status_enabled():
                        print('[TTS] deferred until SPACE released')
                    _schedule_speak_after_release(text, watcher)
                    return
                if _voice_status_enabled():
                    print('[TTS] speaking immediately')
                speak(text)
                return
            if _voice_status_enabled():
                print('[TTS] deferred until SPACE released')
            _schedule_speak_after_release(text, watcher)
            return
    except Exception:
        pass
    if _voice_status_enabled():
        print('[TTS] speaking immediately')
    speak(text)


# -------------------------------- Mic / STT ----------------------------------
class _RecognizerStub:  # minimal surface used in tests (allows monkeypatching .listen)
    def adjust_for_ambient_noise(self, *a, **k):
        return None
    def listen(self, *a, **k):
        raise RuntimeError("recognizer unavailable")
    def recognize_sphinx(self, *a, **k):
        raise RuntimeError("sphinx unavailable")
    def recognize_google(self, *a, **k):
        raise RuntimeError("google recognizer unavailable")

try:
    _recognizer = sr.Recognizer() if sr is not None else _RecognizerStub()
except Exception:  # pragma: no cover
    _recognizer = _RecognizerStub()
_STT_BACKEND = (os.getenv('NERION_STT_BACKEND') or 'auto').strip().lower()
_STT_MODEL = (os.getenv('NERION_STT_MODEL') or 'small').strip().lower()

def set_stt_profile(backend: str | None = None, model: str | None = None) -> None:
    """Set preferred STT backend/model for this session and downstream recognizer."""
    global _STT_BACKEND, _STT_MODEL
    if backend:
        _STT_BACKEND = str(backend).strip().lower()
    if model:
        _STT_MODEL = str(model).strip().lower()
    try:
        _stt_set_profile(_STT_BACKEND, _STT_MODEL)
    except Exception:
        pass


def initial_calibration(seconds: float = 2.0) -> None:
    try:
        with sr.Microphone(device_index=_sr_device_index) as source:
            _recognizer.adjust_for_ambient_noise(source, duration=seconds)
    except Exception:
        pass



def _normalize_vocab(text: Optional[str]) -> Optional[str]:
    """Best-effort post-processing to fix common ASR confusions.

    - Maps common aliases to 'nerion' (e.g., 'marion', 'merion', 'narion').
    - Aliases can be customized via env `NERION_STT_ALIASES` (comma-separated).
    """
    if not text:
        return text
    try:
        s = str(text)
        aliases_env = os.getenv("NERION_STT_ALIASES", "")
        aliases = [a.strip().lower() for a in aliases_env.split(",") if a.strip()]
        if not aliases:
            aliases = [
                "marion", "merion", "marrion", "mary on",
                "neron", "neiron", "narion",
            ]
        for a in aliases:
            s = re.sub(rf"\b{re.escape(a)}\b", "nerion", s, flags=re.I)
        # Common phrase corrections (ASR confusions)
        phrase_map = {
            r"\bself\s+approved\b": "self improve",
            r"\bself[- ]?improved\b": "self improve",
            r"\bself[- ]?approve(d)?\b": "self improve",
            r"\btools\s+plugins\b": "tools-plugins",
            r"\bweb\s+search\b": "web-research",
        }
        for rx, repl in phrase_map.items():
            s = re.sub(rx, repl, s, flags=re.I)
        return s
    except Exception:
        return text


def listen_once(timeout: float = 8.0, phrase_time_limit: float = 6.0) -> Optional[str]:
    try:
        with sr.Microphone(device_index=_sr_device_index) as source:
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except Exception:
        return None
    # Offline-first: prefer Sphinx when NERION_STT_OFFLINE=1 and pocketsphinx is available
    offline = (os.getenv("NERION_STT_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"})
    strict_offline = (os.getenv("NERION_STT_STRICT_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"})
    if offline:
        try:
            # Bias toward 'nerion' as a keyword when available
            txt = _recognizer.recognize_sphinx(audio, keyword_entries=[("nerion", 1.0)])
            txt = (txt or "").lower().strip() or None
            return _normalize_vocab(txt)
        except Exception:
            # In strict offline mode, do not fall back to online recognizers
            if strict_offline:
                return None
            # else fall through to regular path
    try:
        txt = _recognizer.recognize_google(audio).lower().strip()
        return _normalize_vocab(txt)
    except Exception:
        return None


def _mic_frames_ptt(watcher, device_hint: Optional[str], sample_rate: int = 16000, frame_ms: int = 20):
    """Yield PCM16 mono frames while SPACE is held. Uses PyAudio if available."""
    try:
        import pyaudio  # type: ignore
    except Exception:
        return None
    pa = pyaudio.PyAudio()
    try:
        idx = None
        if device_hint:
            try:
                import speech_recognition as _sr
                names = _sr.Microphone.list_microphone_names()
                low = device_hint.lower()
                for i, nm in enumerate(names):
                    if low in (nm or '').lower():
                        idx = i
                        break
            except Exception:
                idx = None
        fmt = pyaudio.paInt16
        ch = 1
        fb = int(sample_rate * (frame_ms / 1000.0))
        stream = pa.open(format=fmt, channels=ch, rate=sample_rate, input=True,
                         input_device_index=idx, frames_per_buffer=fb)
        def _iter():
            try:
                while watcher.space_down.is_set():
                    yield stream.read(fb, exception_on_overflow=False)
            finally:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
                try:
                    pa.terminate()
                except Exception:
                    pass
        return _iter()
    except Exception:
        return None


def _ptt_stream_transcribe(watcher, device_hint: Optional[str]) -> Optional[str]:
    """PTT transcription.

    Preferred path: use streaming recognizer if available.
    Fallback: capture raw PCM frames while SPACE is held, then run single-shot
    recognition on the concatenated audio. This guarantees "hold-to-talk"
    semantics: recording ends exactly on key release.
    """
    frames_iter = _mic_frames_ptt(watcher, device_hint)
    if frames_iter is None:
        return None

    # Fast path with external streaming recognizer
    if transcribe_streaming is not None:
        try:
            def _on_partial(txt: str):
                if txt:
                    print(f"[partial] {txt}")
            t0 = time.time()
            out = transcribe_streaming(
                frames_iter,
                sample_rate=16000,
                language="en",
                model=_STT_MODEL or os.getenv("NERION_STT_MODEL", "small"),
                vad_aggressiveness=int(os.getenv("NERION_VAD_AGGR", "2")),
                min_speech_ms=int(os.getenv("NERION_VAD_MIN_SPEECH", "200")),
                max_silence_ms=int(os.getenv("NERION_VAD_MAX_SILENCE", "800")),
                partial_interval_ms=int(os.getenv("NERION_STT_PARTIAL_MS", "250")),
                partial_cb=_on_partial,
            )
            try:
                dur_ms = int((time.time() - t0) * 1000)
                _log = {
                    'ts': time.time(),
                    'backend': _STT_BACKEND or 'streaming',
                    'model': _STT_MODEL,
                    'duration_ms': dur_ms,
                    'bytes': None,
                }
                os.makedirs(os.path.join('out','voice'), exist_ok=True)
                with open(os.path.join('out','voice','latency.jsonl'), 'a', encoding='utf-8') as f:
                    import json as _json
                    f.write(_json.dumps(_log) + "\n")
            except Exception:
                pass
            return out
        except Exception as e:  # pragma: no cover
            print(f"[STT] streaming failed: {e}")
            # Fall through to buffer-based path

    # Buffer-based path: accumulate frames until release, then recognize
    try:
        buf = bytearray()
        for chunk in frames_iter:
            if chunk:
                buf.extend(chunk)
        if not buf:
            return None
        raw = bytes(buf)
        audio = sr.AudioData(raw, sample_rate=16000, sample_width=2)
        offline = (os.getenv("NERION_STT_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"})
        strict_offline = (os.getenv("NERION_STT_STRICT_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"})
        t0 = time.time()
        if offline:
            try:
                txt = _recognizer.recognize_sphinx(audio)
                out = _normalize_vocab((txt or "").lower().strip() or None)
                return out
            except Exception:
                if strict_offline:
                    return None
        try:
            txt = _recognizer.recognize_google(audio)
            out = _normalize_vocab((txt or "").lower().strip() or None)
            return out
        except Exception:
            return None
        finally:
            try:
                dur_ms = int((time.time() - t0) * 1000)
                _log = {
                    'ts': time.time(),
                    'backend': 'sphinx' if offline else 'google',
                    'model': _STT_MODEL,
                    'duration_ms': dur_ms,
                    'bytes': len(raw),
                }
                os.makedirs(os.path.join('out','voice'), exist_ok=True)
                with open(os.path.join('out','voice','latency.jsonl'), 'a', encoding='utf-8') as f:
                    import json as _json
                    f.write(_json.dumps(_log) + "\n")
            except Exception:
                pass
    except Exception:
        return None


# ------------------------------- Mute control --------------------------------

def set_mute(state: bool) -> None:
    """Mute/unmute using the injected ChatState; speaks a brief confirmation."""
    prev = bool(getattr(_STATE, 'muted', False))
    try:
        _STATE.set_mute(bool(state))
    except Exception:
        return
    if getattr(_STATE, 'muted', False) and not prev:
        print('🔇 Nerion muted.')
        speak("Okay, I'll be quiet.", force=True)
    elif (not getattr(_STATE, 'muted', False)) and prev:
        print('🔊 Nerion unmuted.')
        speak('Voice is back on.', force=True)
def _notify_ui(event_type: str, payload: Optional[dict] = None) -> None:
    try:
        # Prefer direct Electron stdio when running under the Electron launcher
        from . import ipc_electron as _ipc
        if _ipc.enabled():
            _ipc.emit(event_type, payload)
            return
        # Fallback to launching or messaging the Electron subprocess
        from . import ui_bridge
        data = {"type": event_type}
        if payload is not None:
            data["payload"] = payload
        ui_bridge.send_event(data)
    except Exception:
        pass
