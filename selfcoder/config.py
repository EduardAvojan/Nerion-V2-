from __future__ import annotations
import pathlib
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "backups"

_TRUE_SET = {"1", "true", "yes", "on"}
_FALSE_SET = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool = True) -> bool:
    """Parse a boolean from environment variables with sensible defaults.

    Accepts: 1/0, true/false, yes/no, on/off (case-insensitive). When not set or
    unparsable, returns `default`.
    """
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    s = str(val).strip().lower()
    if s in _TRUE_SET:
        return True
    if s in _FALSE_SET:
        return False
    return bool(default)


def allow_network() -> bool:
    """Master switch for *all* network access.

    If `NERION_ALLOW_NETWORK` is unset, defaults to True. Set to 0/false/no/off to
    disable any network activity (URL fetch, augmentation, etc.). Callers should
    check this before performing network I/O.
    """
    return _env_bool("NERION_ALLOW_NETWORK", default=True)


# SAFE mode flag helpers are used by CLI and I/O layers to coordinate simulation vs. real apply.

def self_improve_safe(default: bool = True) -> bool:
    """Return whether self-improve operations should run in SAFE (simulation) mode.

    Environment precedence (case-insensitive booleans: 1/0, true/false, yes/no, on/off):
      1) NERION_SELFIMPROVE_SAFE
      2) NERION_SAFE (fallback umbrella)
      3) default (True unless overridden)
    """
    val = os.environ.get("NERION_SELFIMPROVE_SAFE")
    if val is not None:
        return _env_bool("NERION_SELFIMPROVE_SAFE", default=default)
    # Fallback umbrella toggle
    return _env_bool("NERION_SAFE", default=default)


def is_safe_mode(default: bool = True) -> bool:
    """Alias for self_improve_safe(); provided for symmetry with other modules."""
    return self_improve_safe(default=default)


def get_policy(default: str = 'balanced') -> str:
    """Return the effective runtime policy: 'safe' | 'balanced' | 'fast'."""
    p = (os.environ.get('NERION_POLICY') or '').strip().lower()
    if p in {'safe', 'balanced', 'fast'}:
        return p
    return default
