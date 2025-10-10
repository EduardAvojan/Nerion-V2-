"""pytest configuration for selfcoder tests."""
import pytest
import os


@pytest.fixture(autouse=True)
def disable_governor_and_verification(monkeypatch):
    """Disable governor and verification checks for all tests."""
    # Disable governor to avoid rate limiting
    monkeypatch.setenv("NERION_GOVERNOR_MIN_INTERVAL_MINUTES", "0")
    monkeypatch.setenv("NERION_GOVERNOR_MAX_RUNS_PER_HOUR", "0")
    monkeypatch.setenv("NERION_GOVERNOR_MAX_RUNS_PER_DAY", "0")

    # Disable verification checks to avoid test failures due to missing UI/build infrastructure
    monkeypatch.setenv("NERION_VERIFY_UI_CMD", "skip")
    monkeypatch.setenv("NERION_VERIFY_REG_CMD", "skip")
