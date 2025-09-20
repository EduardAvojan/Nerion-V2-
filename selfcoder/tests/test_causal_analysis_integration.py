from __future__ import annotations
from pathlib import Path

from selfcoder.simulation import run_tests_and_healthcheck
from selfcoder.self_improve import apply as apply_self_improve


def test_simulation_timeouts_produce_analysis():
    """Simulation should attach a structured analysis on timeouts."""
    shadow_root = Path.cwd()
    out = run_tests_and_healthcheck(
        shadow_root,
        skip_pytest=False,
        skip_healthcheck=False,
        pytest_timeout=0,
        healthcheck_timeout=0,
    )

    # Pytest branch
    py = out.get("pytest", {})
    assert isinstance(py, dict)
    assert py.get("rc") == 124
    assert "analysis" in py
    assert py["analysis"].get("root_cause") in {"TimeoutExpired"}

    # Healthcheck branch
    hc = out.get("healthcheck", {})
    assert isinstance(hc, dict)
    assert hc.get("rc") == 124
    assert "analysis" in hc
    assert hc["analysis"].get("root_cause") in {"TimeoutExpired"}


def test_self_improve_apply_json_error_has_analysis(tmp_path):
    """apply() should return structured causal analysis on exceptions."""
    bad = tmp_path / "bad_plan.json"
    bad.write_text("{ not: valid json\n")

    res = apply_self_improve(bad)

    assert res.get("applied") is False
    assert res.get("rolled_back") in {True, False}
    assert "analysis" in res
    # JSON decode typically appears as JSONDecodeError (ValueError subclass).
    # Our analyzer may classify as ValueError root_cause; accept both.
    root = res["analysis"].get("root_cause")
    assert root in {"JSONDecodeError", "ValueError"}