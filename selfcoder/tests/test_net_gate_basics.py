

import pytest

# We will monkeypatch time.time inside the gate module for deterministic tests
from ops.security.net_gate import NetworkGate, NetState


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.t = float(start)

    def time(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


@pytest.fixture(autouse=True)
def reset_gate(monkeypatch):
    """Ensure a clean gate before each test and use a fresh FakeClock."""
    # Fresh fake clock
    fc = FakeClock()
    monkeypatch.setattr("ops.security.net_gate.time.time", fc.time)

    # Re-init the gate with permissive global policy by default
    NetworkGate.init({
        "allow_network_access": True,
        "net": {
            "idle_revoke_after": "15m",
            "remember_by_task_type": True,
        },
        "paths": {},
    })

    yield fc  # tests can advance time via the fixture return value

    # Hard reset between tests
    NetworkGate.revoke(reason="test_teardown")


def test_master_switch_blocks_grant(monkeypatch):
    # Turn off globally
    NetworkGate.init({"allow_network_access": False})
    with pytest.raises(PermissionError):
        NetworkGate.grant_session(minutes=10, reason="should fail")
    assert NetworkGate.state() == NetState.OFFLINE
    assert NetworkGate.can_use("web_search") is False


def test_session_grant_allows(reset_gate):
    NetworkGate.grant_session(minutes=10, reason="allow test")
    assert NetworkGate.state() == NetState.SESSION
    assert NetworkGate.can_use("web_search") is True
    # time_remaining should be within (0, 600]
    tl = NetworkGate.time_remaining()
    assert tl is None or (0 <= tl <= 600.0)  # if idle-timeout is not limiting, None means no cap


def test_sliding_window_refreshes(reset_gate):
    fc: FakeClock = reset_gate
    NetworkGate.grant_session(minutes=10, reason="sliding window")

    # Immediately after grant
    t1 = NetworkGate.time_remaining()
    assert t1 is None or (580.0 <= t1 <= 600.0)

    # Advance 120s; remaining should drop by ~120s
    fc.advance(120)
    t2 = NetworkGate.time_remaining()
    if t1 is not None and t2 is not None:
        assert 430.0 <= t2 <= 480.0  # allow some slack

    # A permitted call should refresh the window back to ~600s
    NetworkGate.assert_allowed(task_type="web_search", url="https://example.com")
    t3 = NetworkGate.time_remaining()
    assert t3 is None or (580.0 <= t3 <= 600.0)


def test_idle_revoke(reset_gate, monkeypatch):
    fc: FakeClock = reset_gate
    # Configure short idle revoke window (5s)
    NetworkGate.init({
        "allow_network_access": True,
        "net": {"idle_revoke_after": "5s"},
    })
    NetworkGate.grant_session(minutes=10, reason="idle test")

    assert NetworkGate.state() == NetState.SESSION
    assert NetworkGate.can_use("web_render", url="https://example.com") is True

    # Advance just under the idle threshold: still allowed
    fc.advance(4.0)
    assert NetworkGate.can_use("web_render", url="https://example.com") is True

    # Cross idle threshold => auto revoke
    fc.advance(2.0)
    # Any state query auto-checks and revokes if needed
    assert NetworkGate.state() == NetState.OFFLINE
    assert NetworkGate.can_use("web_render", url="https://example.com") is False


def test_domain_scoping_when_specified(reset_gate):
    # Grant limited to example.com only
    NetworkGate.grant_session(minutes=10, domains=["example.com"], reason="domain scope")
    # Allowed domain
    NetworkGate.assert_allowed(task_type="web_search", url="https://example.com/path")
    # Different domain should fail
    with pytest.raises(PermissionError):
        NetworkGate.assert_allowed(task_type="web_search", url="https://other.com")