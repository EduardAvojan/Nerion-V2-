import os
from pathlib import Path
from typing import Optional

# Ensure noisy parallel tokenizer warning stays off by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_TRUE = {"1", "true", "yes", "on"}


def get_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return env var as str or *default* if missing/empty."""
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def get_env_bool(name: str, default: bool = False) -> bool:
    """Return env var parsed as boolean with common truthy values."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in _TRUE


def get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def get_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def get_env_path(name: str, default: Path) -> Path:
    """Return a Path from env or the provided default Path."""
    v = get_env_str(name)
    return Path(v) if v else default