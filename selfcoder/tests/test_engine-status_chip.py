

import types
import pytest

# Import the chip helpers from the engine
from app.chat.engine import _status_chip, _fmt_time_left
import app.chat.engine as ENG


class _State:
    def __init__(self, name: str):
        self.name = name


def test_status_chip_offline(monkeypatch):
    # Force NetworkGate.state() -> OFFLINE
    monkeypatch.setattr(ENG.NetworkGate, 'state', staticmethod(lambda: _State('OFFLINE')))
    # time_remaining should not matter offline, but set to a sentinel anyway
    monkeypatch.setattr(ENG.NetworkGate, 'time_remaining', staticmethod(lambda: 0))

    chip = _status_chip()
    assert chip == '🔒 Offline'


def test_status_chip_online_seconds(monkeypatch):
    # SESSION with ~42s left → show seconds
    monkeypatch.setattr(ENG.NetworkGate, 'state', staticmethod(lambda: _State('SESSION')))
    monkeypatch.setattr(ENG.NetworkGate, 'time_remaining', staticmethod(lambda: 42))

    chip = _status_chip()
    assert chip == '🌐 Online (42s left)'


def test_status_chip_online_minutes(monkeypatch):
    # SESSION with 301s left → ceil to 6m
    monkeypatch.setattr(ENG.NetworkGate, 'state', staticmethod(lambda: _State('SESSION')))
    monkeypatch.setattr(ENG.NetworkGate, 'time_remaining', staticmethod(lambda: 301))

    chip = _status_chip()
    assert chip == '🌐 Online (6m left)'


def test_status_chip_online_hours(monkeypatch):
    # SESSION with 3700s → 61.66m → 1h 2m
    monkeypatch.setattr(ENG.NetworkGate, 'state', staticmethod(lambda: _State('SESSION')))
    monkeypatch.setattr(ENG.NetworkGate, 'time_remaining', staticmethod(lambda: 3700))

    chip = _status_chip()
    assert chip == '🌐 Online (1h 2m left)'


def test_fmt_time_left_edge_cases():
    # Directly test formatter edge cases
    assert _fmt_time_left(None) == ''
    assert _fmt_time_left(0) == ' (0s left)'
    assert _fmt_time_left(59) == ' (59s left)'
    assert _fmt_time_left(60) == ' (1m left)'
    assert _fmt_time_left(3599) == ' (60m left)'
    assert _fmt_time_left(3600) == ' (1h left)'
    assert _fmt_time_left(3661) == ' (1h 2m left)'