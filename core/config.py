# core/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import pathlib
import yaml


@dataclass(frozen=True)
class TTSConfig:
    backend: str            # "say" | "pyttsx3"
    rate: int
    device_hint: Optional[str] = None
    profile: Optional[str] = None
    always_speak: bool = True


@dataclass(frozen=True)
class VoiceConfig:
    mode: str               # "ptt" | "always"
    barge_in: bool = False
    vad_sensitivity: int = 8
    vad_min_speech_ms: int = 120
    vad_silence_tail_ms: int = 250


@dataclass(frozen=True)
class MemoryConfig:
    short_term_ttl_days: int = 14
    decay_per_day: float = 0.15
    promotion_threshold: float = 2.5
    max_items: int = 400


@dataclass(frozen=True)
class Config:
    tts: TTSConfig
    voice: VoiceConfig
    memory: MemoryConfig
    build_tag: str = "dev"


def _get(d: dict, path: str, default=None):
    """Safe nested get: _get(data, 'voice.rate', 190)."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_config(settings_path: str = "app/settings.yaml") -> Config:
    data = {}
    p = pathlib.Path(settings_path)
    if p.exists():
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    # Environment overrides (non-invasive)
    env_backend = os.getenv("NERION_TTS_BACKEND")
    env_rate = os.getenv("NERION_TTS_RATE")

    tts_backend = (env_backend or _get(data, "voice.tts_backend", "say")).strip().lower()
    tts_rate = int(env_rate) if env_rate is not None else int(_get(data, "voice.rate", 190))

    tts = TTSConfig(
        backend=tts_backend,
        rate=tts_rate,
        device_hint=_get(data, "voice.device", None),
        profile=_get(data, "voice.profile", None),
        always_speak=bool(_get(data, "voice.always_speak", True)),
    )
    voice = VoiceConfig(
        mode=str(_get(data, "voice.mode", "ptt")),
        barge_in=bool(_get(data, "voice.barge_in", False)),
        vad_sensitivity=int(_get(data, "voice.vad.sensitivity", 8)),
        vad_min_speech_ms=int(_get(data, "voice.vad.min_speech_ms", 120)),
        vad_silence_tail_ms=int(_get(data, "voice.vad.silence_tail_ms", 250)),
    )
    memory = MemoryConfig(
        short_term_ttl_days=int(_get(data, "memory.short_term_ttl_days", 14)),
        decay_per_day=float(_get(data, "memory.decay_per_day", 0.15)),
        promotion_threshold=float(_get(data, "memory.promotion_threshold", 2.5)),
        max_items=int(_get(data, "memory.max_items", 400)),
    )
    build_tag = str(_get(data, "build_tag", _get(data, "BUILD_TAG", "dev")))

    return Config(tts=tts, voice=voice, memory=memory, build_tag=build_tag)