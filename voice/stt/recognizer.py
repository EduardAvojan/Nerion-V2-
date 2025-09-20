from __future__ import annotations

"""
Lightweight offline STT adapter with optional Whisper/Vosk backends.

Exports:
- transcribe_streaming(frames_iter, sample_rate, language, model, vad_aggressiveness, ...)
- set_profile(backend, model)
- start_barge_in_monitor(on_speech, device=None, sensitivity=2, min_speech_ms=200, silence_tail_ms=800)
- stop_barge_in_monitor()
- run_diagnostics(...): best‑effort mic/VAD check returning True on success

Behavior:
- If backend is 'whisper' and whisper is importable, transcribes buffered audio.
- Else if backend is 'vosk' and vosk is importable with a default model, uses KaldiRecognizer.
- Else falls back to pocketsphinx via SpeechRecognition if available.
- Returns a final transcript string. Partial callbacks are best-effort no-ops.

No network calls. This module is intentionally conservative: it does not attempt
to download models; users should install and configure models separately.
"""

import io
import os
from typing import Callable, Optional

_BACKEND = os.getenv("NERION_STT_BACKEND", "auto").strip().lower() or "auto"
_MODEL = os.getenv("NERION_STT_MODEL", "small").strip().lower() or "small"
_WARNED_MISSING = {
    'whispercpp': False,
}


def set_profile(backend: Optional[str] = None, model: Optional[str] = None) -> None:
    global _BACKEND, _MODEL
    if backend:
        _BACKEND = backend.strip().lower()
    if model:
        _MODEL = model.strip().lower()
    # Env validation warnings (once)
    try:
        if _BACKEND in {'whisper.cpp', 'whispercpp'}:
            if not (os.getenv('WHISPER_CPP_MODEL_PATH') or os.getenv('NERION_WHISPER_CPP_MODEL')):
                if not _WARNED_MISSING['whispercpp']:
                    print('[STT:whisper.cpp] warning: WHISPER_CPP_MODEL_PATH not set (local model required)')
                    _WARNED_MISSING['whispercpp'] = True
    except Exception:
        pass


def _as_wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    import wave
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16)
    return bio.getvalue()


def _try_whisper(pcm16: bytes, sample_rate: int, language: str, model: str) -> Optional[str]:
    try:
        import numpy as np  # type: ignore
        import whisper  # type: ignore
        # Convert PCM16 to float32 waveform
        arr = np.frombuffer(pcm16, dtype=np.int16).astype('float32') / 32768.0
        # Whisper expects 16k; resample only if needed (skip to avoid heavy deps)
        if int(sample_rate) != 16000:
            # crude nearest-neighbor resample to 16k to avoid bringing in scipy
            ratio = 16000.0 / float(sample_rate)
            idx = (np.arange(int(len(arr) * ratio)) / ratio).astype(int)
            arr = arr[idx]
        m = whisper.load_model(model)
        out = m.transcribe(arr, language=language or 'en', fp16=False)
        text = (out or {}).get('text') if isinstance(out, dict) else None
        return str(text).strip() if text else None
    except Exception:
        return None


def _try_vosk(pcm16: bytes, sample_rate: int, language: str, model: str) -> Optional[str]:
    # Vosk requires a model directory on disk; we only run if VOSK_MODEL is set
    try:
        import json
        from vosk import Model, KaldiRecognizer  # type: ignore
        model_dir = os.getenv('VOSK_MODEL')
        if not model_dir:
            return None
        m = Model(model_dir)
        rec = KaldiRecognizer(m, int(sample_rate))
        # Feed in one chunk; recognizer accumulates internally
        rec.AcceptWaveform(pcm16)
        data = json.loads(rec.Result() or '{}')
        txt = data.get('text') or ''
        return txt.strip() or None
    except Exception:
        return None

def _try_whispercpp(pcm16: bytes, sample_rate: int, language: str) -> Optional[str]:
    """Best-effort whisper.cpp Python binding; requires local GGML/GGUF and env WHISPER_CPP_MODEL_PATH.

    This path writes a temp WAV and calls the library if present; otherwise returns None.
    """
    model_path = os.getenv('WHISPER_CPP_MODEL_PATH') or os.getenv('NERION_WHISPER_CPP_MODEL')
    if not model_path:
        return None
    try:
        # Try two popular bindings
        wlib = None
        try:
            import whispercpp  # type: ignore
            wlib = ('whispercpp', whispercpp)
        except Exception:
            try:
                import whisper_cpp as whispercpp2  # type: ignore
                wlib = ('whisper_cpp', whispercpp2)
            except Exception:
                wlib = None
        if wlib is None:
            return None
        # Write to WAV and transcribe
        wav = _as_wav_bytes(pcm16, sample_rate)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(wav)
            wav_path = tmp.name
        txt = None
        try:
            name, lib = wlib
            # The exact API varies; try common patterns
            if hasattr(lib, 'Whisper'):  # e.g., whispercpp.Whisper
                try:
                    model = getattr(lib, 'Whisper')(model_path)
                    res = model.transcribe(wav_path, language=language or 'en')
                    txt = str(res).strip() if res is not None else None
                except Exception:
                    txt = None
            elif hasattr(lib, 'Model'):
                try:
                    m = getattr(lib, 'Model')(model_path)
                    out = m.transcribe(wav_path)
                    txt = str(out) if out else None
                except Exception:
                    txt = None
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass
        return txt if txt else None
    except Exception:
        return None


def _try_sphinx(pcm16: bytes, sample_rate: int, language: str) -> Optional[str]:
    try:
        import speech_recognition as sr  # type: ignore
        audio = sr.AudioData(pcm16, sample_rate=int(sample_rate), sample_width=2)
        rec = sr.Recognizer()
        try:
            # allow keyword to bias towards wake-word spelling
            return (rec.recognize_sphinx(audio, language=language) or '').strip() or None
        except TypeError:
            return (rec.recognize_sphinx(audio) or '').strip() or None
    except Exception:
        return None


def transcribe_streaming(
    frames_iter,
    *,
    sample_rate: int,
    language: str = 'en',
    model: str = 'small',
    vad_aggressiveness: int = 2,
    min_speech_ms: int = 200,
    max_silence_ms: int = 800,
    partial_interval_ms: int = 250,
    partial_cb: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Consume frames until iterator ends and return a final transcript.

    This is a buffered implementation to avoid heavy streaming complexity.
    """
    try:
        buf = bytearray()
        for chunk in frames_iter:
            if chunk:
                buf.extend(chunk)
        pcm = bytes(buf)
        backend = _BACKEND if _BACKEND != 'auto' else (os.getenv('NERION_STT_BACKEND') or 'sphinx')
        backend = backend.strip().lower()
        # Whisper path
        if backend == 'whisper':
            txt = _try_whisper(pcm, sample_rate, language, model or _MODEL)
            if txt:
                return txt
            # fall back to sphinx if whisper failed
        # whisper.cpp path
        if backend in {'whisper.cpp', 'whispercpp'}:
            txt = _try_whispercpp(pcm, sample_rate, language)
            if txt:
                return txt
        # Vosk path
        if backend == 'vosk':
            txt = _try_vosk(pcm, sample_rate, language, model or _MODEL)
            if txt:
                return txt
        # Default sphinx
        txt = _try_sphinx(pcm, sample_rate, language)
        return txt
    except Exception:
        return None


# ------------------------ Barge‑in monitor (VAD) ----------------------------

_barge_thread = None
_barge_stop = None


def _resolve_input_device_index_by_name(name_substr: Optional[str]) -> Optional[int]:
    try:
        import pyaudio  # type: ignore
        pa = pyaudio.PyAudio()
        try:
            if not name_substr:
                return None
            target = (name_substr or '').strip().lower()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                nm = str(info.get('name') or '').lower()
                if target and target in nm and int(info.get('maxInputChannels') or 0) > 0:
                    return int(i)
        finally:
            try:
                pa.terminate()
            except Exception:
                pass
    except Exception:
        return None
    return None


def start_barge_in_monitor(
    on_speech: Callable[[], None],
    *,
    device: Optional[str] = None,
    sensitivity: int = 2,
    min_speech_ms: int = 200,
    silence_tail_ms: int = 800,
    sample_rate: int = 16000,
    frame_ms: int = 20,
) -> bool:
    """Start a lightweight background VAD monitor that calls on_speech() on speech onset.

    Best‑effort: uses PyAudio + webrtcvad (if available) or energy gate fallback.
    Returns True if the monitor started, False otherwise.
    """
    global _barge_thread, _barge_stop
    if _barge_thread and _barge_thread.is_alive():
        return True
    try:
        import pyaudio  # type: ignore
        from voice.stt.vad import segment_stream  # local import
    except Exception:
        return False

    dev_index = _resolve_input_device_index_by_name(device)
    _barge_stop = _barge_stop or __import__('threading').Event()
    _barge_stop.clear()

    def _frames():
        pa = pyaudio.PyAudio()
        try:
            fb = int(sample_rate * (frame_ms / 1000.0)) * 2
            kwargs = dict(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=fb)
            if dev_index is not None:
                kwargs['input_device_index'] = dev_index
            stream = pa.open(**kwargs)
            try:
                while _barge_stop and not _barge_stop.is_set():
                    yield stream.read(fb, exception_on_overflow=False)
            finally:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
        finally:
            try:
                pa.terminate()
            except Exception:
                pass

    def _run():
        try:
            for seg in segment_stream(
                _frames(),
                sample_rate=sample_rate,
                aggressiveness=int(sensitivity),
                min_speech_ms=int(min_speech_ms),
                max_silence_ms=int(silence_tail_ms),
                frame_ms=int(frame_ms),
            ):
                try:
                    if callable(on_speech):
                        on_speech()
                except Exception:
                    pass
                # throttle to avoid repeated triggers
                __import__('time').sleep(0.3)
                if _barge_stop and _barge_stop.is_set():
                    break
        except Exception:
            # swallow errors to keep UI stable
            pass

    th = __import__('threading').Thread(target=_run, name='nerion_barge_in', daemon=True)
    _barge_thread = th
    try:
        th.start()
        return True
    except Exception:
        return False


def stop_barge_in_monitor() -> None:
    """Stop the background barge‑in monitor if running."""
    global _barge_thread, _barge_stop
    try:
        if _barge_stop:
            _barge_stop.set()
    except Exception:
        pass
    try:
        if _barge_thread and _barge_thread.is_alive():
            _barge_thread.join(timeout=0.5)
    except Exception:
        pass
    _barge_thread = None


# ------------------------ Diagnostics (best‑effort) -------------------------

def run_diagnostics(
    *,
    duration: int = 5,
    device: Optional[str] = None,
    vad_sensitivity: Optional[int] = None,
    min_speech_ms: Optional[int] = None,
    silence_tail_ms: Optional[int] = None,
    color: bool = True,
) -> bool:
    """Best‑effort mic/VAD diagnostics. Returns True if basic checks pass.

    This avoids heavy dependencies; if audio stack is unavailable, returns False.
    """
    try:
        ok = start_barge_in_monitor(lambda: None, device=device, sensitivity=int(vad_sensitivity or 2), min_speech_ms=int(min_speech_ms or 200), silence_tail_ms=int(silence_tail_ms or 800))
        if not ok:
            return False
        # Run briefly then stop
        __import__('time').sleep(max(1, int(duration)))
        stop_barge_in_monitor()
        return True
    except Exception:
        return False
