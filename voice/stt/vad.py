from __future__ import annotations

"""Lightweight streaming VAD for 16kHz mono PCM16 audio.

This module segments an incoming stream of small PCM16 frames (e.g., 20ms) into
complete speech utterances. It uses WebRTC VAD if available; otherwise a tiny
energy-gate fallback. No network, no heavy deps.

Usage
-----
frames = mic_frame_generator()  # yields bytes of PCM16 mono @16kHz, ~20ms each
for seg in segment_stream(frames, sample_rate=16000):
    # seg is a bytes object of concatenated PCM16 frames representing one utterance
    do_stt(seg)
"""

from typing import Iterable, Iterator, List

_VAD_AVAILABLE = False
try:  # optional dependency
    import webrtcvad  # type: ignore
    _VAD_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    webrtcvad = None  # type: ignore


def _frame_bytes(sample_rate: int, frame_ms: int) -> int:
    """Number of bytes for a PCM16 mono frame of length frame_ms at sample_rate."""
    return int(sample_rate * (frame_ms / 1000.0)) * 2


def _energy_gt(pcm: bytes, thr: float = 0.01) -> bool:
    """Simple RMS energy gate in lieu of VAD. pcm is PCM16 mono bytes."""
    if not pcm:
        return False
    try:
        import numpy as np  # local import to avoid hard dependency on import
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float((x * x).mean() ** 0.5)
        return rms > thr
    except Exception:  # pragma: no cover - numpy missing or other
        # Very conservative fallback: treat everything as non-speech
        return False


def segment_stream(
    frames: Iterable[bytes],
    *,
    sample_rate: int = 16000,
    aggressiveness: int = 2,
    min_speech_ms: int = 200,
    max_silence_ms: int = 800,
    frame_ms: int = 20,
) -> Iterator[bytes]:
    """Yield finalized speech segments (PCM16) from a stream of PCM16 frames.

    Parameters
    ----------
    frames : Iterable[bytes]
        An iterable of PCM16 mono frames (ideally frame_ms each). If a frame is
        larger, it will be split into exact frame-sized chunks; smaller chunks
        are dropped.
    sample_rate : int
        Sample rate of PCM16 audio. Default: 16000.
    aggressiveness : int
        WebRTC VAD aggressiveness (0–3). Ignored if webrtcvad not installed.
    min_speech_ms : int
        Minimum amount of speech before we consider an utterance started.
    max_silence_ms : int
        Amount of continuous silence that ends an utterance.
    frame_ms : int
        Frame size in milliseconds. Default: 20ms.
    """
    import time

    # Initialize VAD if available
    vad = None
    if _VAD_AVAILABLE:
        try:
            vad = webrtcvad.Vad(max(0, min(3, int(aggressiveness))))
        except Exception:
            vad = None

    fb = _frame_bytes(sample_rate, frame_ms)

    def is_speech(pcm: bytes) -> bool:
        if vad is not None:
            try:
                return vad.is_speech(pcm, sample_rate)
            except Exception:
                return False
        return _energy_gt(pcm)

    buf: List[bytes] = []
    speech_started = False
    speech_ms = 0
    last_speech_ts = 0.0

    for raw in frames:
        if not raw:
            continue
        # split to exact frame size
        for off in range(0, len(raw), fb):
            chunk = raw[off : off + fb]
            if len(chunk) < fb:
                continue
            if is_speech(chunk):
                buf.append(chunk)
                if not speech_started:
                    speech_started = True
                    speech_ms = 0
                speech_ms += frame_ms
                last_speech_ts = time.time()
            else:
                # end if we have speech and enough trailing silence
                if speech_started and (time.time() - last_speech_ts) * 1000.0 >= max_silence_ms:
                    if speech_ms >= min_speech_ms and buf:
                        yield b"".join(buf)
                    # reset
                    buf.clear()
                    speech_started = False
                    speech_ms = 0

    # flush at end of stream
    if speech_started and speech_ms >= min_speech_ms and buf:
        yield b"".join(buf)
