"""Central governor for autonomous execution controls.

This module enforces rate limiting, scheduling windows, and manual override
hooks for self-directed apply flows (self-improve, planner auto-apply, voice
self coding, upgrade agent, etc.).

Configuration sources (lowest precedence first):
  • built-in defaults (safe/balanced)
  • config/governor.yaml if present
  • environment overrides (NERION_GOVERNOR_*)

State is persisted under out/governor/state.json so we can enforce sliding
window limits without hitting the telemetry store for every check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml optional for tests
    yaml = None  # type: ignore

from ops.security.fs_guard import ensure_in_repo_auto

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_STATE_PATH = Path("out/governor/state.json")
_CONFIG_PATH = Path("config/governor.yaml")
_HISTORY_RETENTION_DAYS = 14  # keep two weeks of governor history

_DEFAULT_CONFIG = {
    "min_interval_minutes": 30,
    "max_runs": {
        "hour": 2,
        "day": 6,
    },
    "windows": [],  # e.g., ["09:00-18:00", "21:00-23:00"]; empty means 24/7
}

_OVERRIDE_ENVS = (
    "NERION_GOVERNOR_OVERRIDE",
    "NERION_SELF_IMPROVE_FORCE",
    "NERION_PLAN_FORCE_GOVERNOR",
    "NERION_UPGRADE_FORCE",
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GovernorDecision:
    """Outcome of a governor check."""

    allowed: bool
    code: str  # ok | override | window | rate_limit
    reasons: List[str] = field(default_factory=list)
    override_used: bool = False
    next_allowed_utc: Optional[str] = None
    next_allowed_local: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def is_blocked(self) -> bool:
        return not self.allowed


# ---------------------------------------------------------------------------
# Helpers: persistence
# ---------------------------------------------------------------------------


def _state_file() -> Path:
    return ensure_in_repo_auto(_STATE_PATH)


def _config_file() -> Path:
    return ensure_in_repo_auto(_CONFIG_PATH)


def _load_state() -> Dict[str, List[str]]:
    path = _state_file()
    try:
        if not path.exists():
            return {"operations": {}}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"operations": {}}
        ops = data.get("operations")
        if isinstance(ops, dict):
            return {"operations": {str(k): list(v) for k, v in ops.items() if isinstance(v, list)}}
    except Exception:
        pass
    return {"operations": {}}


def _save_state(state: Dict[str, List[str]]) -> None:
    path = _state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_yaml_config() -> Dict[str, object]:
    cfg_path = _config_file()
    if yaml is None or not cfg_path.exists():
        return {}
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Helpers: parsing + math
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_list(name: str) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_time_token(token: str) -> int:
    parts = token.strip().split(":")
    if not parts or len(parts) > 2:
        raise ValueError(f"invalid time token: {token}")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) == 2 else 0
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"invalid time value: {token}")
    return hour * 60 + minute


def _parse_windows(entries: Iterable[str]) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for entry in entries or []:
        token = entry.strip()
        if not token:
            continue
        if token in {"*", "all", "ANY"}:
            return [(0, 1440)]
        if "-" not in token:
            continue
        start_raw, end_raw = token.split("-", 1)
        try:
            start = _parse_time_token(start_raw)
            end = _parse_time_token(end_raw)
        except ValueError:
            continue
        windows.append((start, end))
    return windows


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _within_window(minute: int, window: Tuple[int, int]) -> bool:
    start, end = window
    if start == end:
        return False  # zero-width window
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end  # overnight window


def _minutes_until_window(minute: int, window: Tuple[int, int]) -> int:
    start, end = window
    if _within_window(minute, window):
        return 0
    if start < end:
        if minute < start:
            return start - minute
        return (start + 1440) - minute
    # overnight case: we are guaranteed minute not in window ⇒ end ≤ minute < start
    return start - minute


def _build_config() -> Dict[str, object]:
    cfg = dict(_DEFAULT_CONFIG)
    file_cfg = _load_yaml_config()
    if file_cfg:
        cfg.update({k: v for k, v in file_cfg.items() if k in cfg})
        if isinstance(file_cfg.get("max_runs"), dict):
            max_runs = cfg.setdefault("max_runs", {}).copy()
            max_runs.update({k: int(v) for k, v in file_cfg["max_runs"].items() if isinstance(v, int)})
            cfg["max_runs"] = max_runs
        if isinstance(file_cfg.get("windows"), (list, tuple)):
            cfg["windows"] = [str(x) for x in file_cfg.get("windows", [])]
    # Environment overrides
    cfg["min_interval_minutes"] = _env_int(
        "NERION_GOVERNOR_MIN_INTERVAL_MINUTES",
        int(cfg.get("min_interval_minutes", 0) or 0),
    )
    max_runs = dict(cfg.get("max_runs") or {})
    max_runs["hour"] = _env_int(
        "NERION_GOVERNOR_MAX_RUNS_PER_HOUR",
        int(max_runs.get("hour", 0) or 0),
    )
    max_runs["day"] = _env_int(
        "NERION_GOVERNOR_MAX_RUNS_PER_DAY",
        int(max_runs.get("day", 0) or 0),
    )
    cfg["max_runs"] = max_runs
    env_windows = _env_list("NERION_GOVERNOR_WINDOWS")
    if env_windows:
        cfg["windows"] = env_windows
    cfg["windows"] = [str(x) for x in (cfg.get("windows") or [])]
    return cfg


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def _collect_history(operation: str, now: datetime) -> List[datetime]:
    state = _load_state()
    entries = state.get("operations", {}).get(operation, [])
    history: List[datetime] = []
    cutoff = now - timedelta(days=_HISTORY_RETENTION_DAYS)
    for ts in entries:
        dt = _parse_iso(ts)
        if not dt:
            continue
        if dt >= cutoff:
            history.append(dt)
    history.sort()
    return history


def _store_history(operation: str, history: List[datetime]) -> None:
    state = _load_state()
    ops = state.setdefault("operations", {})
    ops[operation] = [dt.isoformat().replace("+00:00", "Z") for dt in history]
    _save_state(state)


def note_execution(operation: str, *, when: Optional[datetime] = None) -> None:
    """Record a successful autonomous execution for rate limiting."""
    now = when or _now_utc()
    history = _collect_history(operation, now)
    history.append(now)
    _store_history(operation, history)


def _override_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)
    for name in _OVERRIDE_ENVS:
        raw = os.getenv(name)
        if raw and raw.strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def evaluate(
    operation: str,
    *,
    now: Optional[datetime] = None,
    override: Optional[bool] = None,
) -> GovernorDecision:
    """Evaluate whether *operation* may proceed under current governor settings."""

    if not operation:
        return GovernorDecision(allowed=True, code="ok")

    timestamp = now or _now_utc()
    cfg = _build_config()

    if _override_enabled(override):
        return GovernorDecision(
            allowed=True,
            code="override",
            override_used=True,
            reasons=["Manual override flag present"],
        )

    # Scheduling windows (local time)
    windows = _parse_windows(cfg.get("windows", []))
    local_now = timestamp.astimezone()
    local_minute = local_now.hour * 60 + local_now.minute
    if windows and not any(_within_window(local_minute, w) for w in windows):
        deltas = [_minutes_until_window(local_minute, w) for w in windows]
        delta_min = min([d for d in deltas if d >= 0], default=None)
        next_allowed_dt = None
        if delta_min is not None:
            next_allowed_dt = local_now + timedelta(minutes=delta_min)
        utc_next = next_allowed_dt.astimezone(timezone.utc) if next_allowed_dt else None
        return GovernorDecision(
            allowed=False,
            code="window",
            reasons=["Outside permitted execution window"],
            next_allowed_local=next_allowed_dt.isoformat() if next_allowed_dt else None,
            next_allowed_utc=utc_next.isoformat().replace("+00:00", "Z") if utc_next else None,
            metadata={"operation": operation},
        )

    # Rate limiting
    history = _collect_history(operation, timestamp)
    reasons: List[str] = []
    next_allowed_candidates: List[datetime] = []

    min_interval = max(0, int(cfg.get("min_interval_minutes") or 0))
    if min_interval and history:
        last = history[-1]
        delta = timestamp - last
        if delta < timedelta(minutes=min_interval):
            wait_until = last + timedelta(minutes=min_interval)
            reasons.append(
                f"Minimum interval {min_interval}m not satisfied (last at {last.isoformat()})"
            )
            next_allowed_candidates.append(wait_until)

    max_runs_cfg = cfg.get("max_runs") or {}

    hour_limit = int(max_runs_cfg.get("hour") or 0)
    if hour_limit > 0:
        recent = [dt for dt in history if timestamp - dt < timedelta(hours=1)]
        if len(recent) >= hour_limit:
            boundary = recent[-hour_limit]
            wait_until = boundary + timedelta(hours=1)
            reasons.append(
                f"Reached hourly cap ({hour_limit} runs per hour)"
            )
            next_allowed_candidates.append(wait_until)

    day_limit = int(max_runs_cfg.get("day") or 0)
    if day_limit > 0:
        recent_day = [dt for dt in history if timestamp - dt < timedelta(days=1)]
        if len(recent_day) >= day_limit:
            boundary = recent_day[-day_limit]
            wait_until = boundary + timedelta(days=1)
            reasons.append(
                f"Reached daily cap ({day_limit} runs per day)"
            )
            next_allowed_candidates.append(wait_until)

    if reasons:
        next_allowed_dt = max(next_allowed_candidates) if next_allowed_candidates else None
        local_next = next_allowed_dt.astimezone() if next_allowed_dt else None
        utc_next = next_allowed_dt.astimezone(timezone.utc) if next_allowed_dt else None
        return GovernorDecision(
            allowed=False,
            code="rate_limit",
            reasons=reasons,
            next_allowed_local=local_next.isoformat() if local_next else None,
            next_allowed_utc=utc_next.isoformat().replace("+00:00", "Z") if utc_next else None,
            metadata={
                "operation": operation,
                "history_count": len(history),
            },
        )

    return GovernorDecision(
        allowed=True,
        code="ok",
        metadata={"operation": operation, "history_count": len(history)},
    )


__all__ = [
    "GovernorDecision",
    "evaluate",
    "note_execution",
]
