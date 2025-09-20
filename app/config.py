"""Central configuration loader.

Precedence: settings.yaml → environment (NERION_*) → (future) CLI.
Env parsing delegates to ops.runtime.env helpers for consistent typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Dict
import json

from ops.runtime import env as envutil

try:  # Optional dependency in tests
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# -----------------------------
# dataclasses
# -----------------------------

@dataclass
class VoiceConfig:
    backend: str = "auto"  # "say" | "pyttsx3" | "auto"
    device: Optional[str] = None
    mode: str = "ptt"  # "ptt" or "open"
    always_speak: bool = True
    rate: float = 0.95  # human-ish speech rate multiplier (macOS say)
    profile: str = "british_male_1"
    barge_in: bool = False
    short_replies: bool = False


@dataclass
class MemoryConfig:
    short_term_ttl_days: int = 14
    decay_per_day: float = 0.15
    promotion_threshold: float = 2.5
    max_items: int = 400
    memory_db_path: Path = Path("memory_db.json")
    journal_path: Path = Path("app/journal.jsonl")


@dataclass
class PathsConfig:
    root_dir: Path = Path.cwd()
    settings_path: Path = Path("app/settings.yaml")


@dataclass
class FeaturesConfig:
    enable_self_coding: bool = True
    enable_voice: bool = True


@dataclass
class Config:
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# loading / singleton access
# -----------------------------

_CONFIG: Optional[Config] = None


def _load_yaml_settings(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        if yaml is None:
            # Allow JSON as a fallback (useful in some test sandboxes)
            return json.loads(path.read_text(encoding="utf-8"))
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _merge_voice_from_env(v: VoiceConfig) -> VoiceConfig:
    v.backend = (envutil.get_env_str("NERION_TTS_BACKEND", v.backend) or v.backend).strip()
    if envutil.get_env_bool("NERION_FORCE_SAY", False):  # legacy flag
        v.backend = "say"
    v.device = envutil.get_env_str("NERION_VOICE_DEVICE", v.device) or v.device
    v.mode = envutil.get_env_str("NERION_SPEECH_MODE", v.mode) or v.mode
    v.always_speak = envutil.get_env_bool("NERION_ALWAYS_SPEAK", v.always_speak)
    v.rate = envutil.get_env_float("NERION_VOICE_RATE", v.rate)
    v.profile = envutil.get_env_str("NERION_VOICE_PROFILE", v.profile) or v.profile
    v.barge_in = envutil.get_env_bool("NERION_BARGE_IN", v.barge_in)
    v.short_replies = envutil.get_env_bool("NERION_SHORT_REPLIES", v.short_replies)
    return v


def _merge_memory_from_env(m: MemoryConfig) -> MemoryConfig:
    m.short_term_ttl_days = envutil.get_env_int("NERION_MEM_TTL_DAYS", m.short_term_ttl_days)
    m.decay_per_day = envutil.get_env_float("NERION_MEM_DECAY_PER_DAY", m.decay_per_day)
    m.promotion_threshold = envutil.get_env_float("NERION_MEM_PROMOTION_THRESHOLD", m.promotion_threshold)
    m.max_items = envutil.get_env_int("NERION_MEM_MAX_ITEMS", m.max_items)
    m.journal_path = envutil.get_env_path("NERION_JOURNAL_PATH", m.journal_path)
    m.memory_db_path = envutil.get_env_path("NERION_MEMORY_DB_PATH", m.memory_db_path)
    return m


def _merge_features_from_env(f: FeaturesConfig) -> FeaturesConfig:
    f.enable_self_coding = envutil.get_env_bool("NERION_ENABLE_SELF_CODING", f.enable_self_coding)
    f.enable_voice = envutil.get_env_bool("NERION_ENABLE_VOICE", f.enable_voice)
    return f


def _apply_settings_dict(cfg: Config, data: Dict[str, Any]) -> Config:
    # Accept both nested and flat keys; be forgiving.
    voice = data.get("voice", {}) or {}
    memory = data.get("memory", {}) or {}
    paths = data.get("paths", {}) or {}
    features = data.get("features", {}) or {}

    # voice
    for k in ("backend", "device", "mode", "always_speak", "rate", "profile", "barge_in", "short_replies"):
        if k in voice:
            setattr(cfg.voice, k, voice[k])

    # memory
    for k in ("short_term_ttl_days", "decay_per_day", "promotion_threshold", "max_items"):
        if k in memory:
            setattr(cfg.memory, k, memory[k])
    if "journal_path" in memory:
        cfg.memory.journal_path = Path(memory["journal_path"])  # type: ignore[arg-type]
    if "memory_db_path" in memory:
        cfg.memory.memory_db_path = Path(memory["memory_db_path"])  # type: ignore[arg-type]

    # paths
    if "root_dir" in paths:
        cfg.paths.root_dir = Path(paths["root_dir"])  # type: ignore[arg-type]
    if "settings_path" in paths:
        cfg.paths.settings_path = Path(paths["settings_path"])  # type: ignore[arg-type]

    # features
    for k in ("enable_self_coding", "enable_voice"):
        if k in features:
            setattr(cfg.features, k, features[k])

    return cfg


def load_config(settings_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML + env overrides.

    This function is pure (no module-level state). Use :func:`get_config`
    if you want the cached singleton for general app use.
    """
    cfg = Config()

    # Where to read settings
    if settings_path is None:
        # Prefer app/settings.yaml if it exists relative to project root
        default_path = Path("app/settings.yaml")
        settings_path = default_path if default_path.exists() else cfg.paths.settings_path

    # YAML (or JSON fallback) → cfg
    data = _load_yaml_settings(settings_path)
    cfg = _apply_settings_dict(cfg, data)

    # ENV overrides → cfg
    cfg.voice = _merge_voice_from_env(cfg.voice)
    cfg.memory = _merge_memory_from_env(cfg.memory)
    cfg.features = _merge_features_from_env(cfg.features)

    return cfg


def get_config() -> Config:
    """Return a cached Config instance (singleton-ish).

    Safe to call anywhere in the app. Tests can call :func:`reload_config`
    to reset.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def reload_config(settings_path: Optional[Path] = None) -> Config:
    """Force reload configuration and update the cache.

    Useful in tests and live-reload workflows.
    """
    global _CONFIG
    _CONFIG = load_config(settings_path)
    return _CONFIG