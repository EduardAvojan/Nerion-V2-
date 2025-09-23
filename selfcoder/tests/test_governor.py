from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import selfcoder.governor as governor


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(governor, "_STATE_PATH", state_path)
    monkeypatch.setattr(governor, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(governor, "ensure_in_repo_auto", lambda path: path)
    for name in (
        "NERION_GOVERNOR_MIN_INTERVAL_MINUTES",
        "NERION_GOVERNOR_MAX_RUNS_PER_HOUR",
        "NERION_GOVERNOR_MAX_RUNS_PER_DAY",
        "NERION_GOVERNOR_WINDOWS",
        "NERION_GOVERNOR_OVERRIDE",
        "NERION_SELF_IMPROVE_FORCE",
        "NERION_PLAN_FORCE_GOVERNOR",
        "NERION_UPGRADE_FORCE",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2025, 1, 1, hour, minute, tzinfo=timezone.utc)


def test_evaluate_allows_without_limits():
    decision = governor.evaluate("test.op", now=_dt(12))
    assert decision.allowed
    assert decision.code == "ok"


def test_min_interval_blocks(monkeypatch):
    monkeypatch.setenv("NERION_GOVERNOR_MIN_INTERVAL_MINUTES", "10")
    base = _dt(10)
    governor.note_execution("test.op", when=base)
    decision = governor.evaluate("test.op", now=base + timedelta(minutes=5))
    assert decision.is_blocked()
    assert decision.code == "rate_limit"
    assert any("Minimum interval" in msg for msg in decision.reasons)


def test_hourly_cap(monkeypatch):
    monkeypatch.setenv("NERION_GOVERNOR_MIN_INTERVAL_MINUTES", "0")
    monkeypatch.setenv("NERION_GOVERNOR_MAX_RUNS_PER_HOUR", "2")
    now = _dt(15)
    governor.note_execution("test.op", when=now - timedelta(minutes=50))
    governor.note_execution("test.op", when=now - timedelta(minutes=10))
    decision = governor.evaluate("test.op", now=now)
    assert decision.is_blocked()
    assert decision.code == "rate_limit"
    assert any("hourly cap" in msg for msg in decision.reasons)


def test_daily_cap(monkeypatch):
    monkeypatch.setenv("NERION_GOVERNOR_MAX_RUNS_PER_DAY", "1")
    base = _dt(8)
    governor.note_execution("test.op", when=base)
    decision = governor.evaluate("test.op", now=base + timedelta(hours=12))
    assert decision.is_blocked()
    assert decision.code == "rate_limit"


def test_windows_block(monkeypatch):
    monkeypatch.setenv("NERION_GOVERNOR_WINDOWS", "12:00-13:00")
    decision = governor.evaluate("test.op", now=_dt(14))
    assert decision.is_blocked()
    assert decision.code == "window"
    assert decision.next_allowed_local is not None


def test_override_via_env(monkeypatch):
    monkeypatch.setenv("NERION_GOVERNOR_WINDOWS", "12:00-13:00")
    monkeypatch.setenv("NERION_GOVERNOR_OVERRIDE", "1")
    decision = governor.evaluate("test.op", now=_dt(14))
    assert decision.allowed
    assert decision.code == "override"
    assert decision.override_used
