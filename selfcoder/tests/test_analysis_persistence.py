from __future__ import annotations
from pathlib import Path
import time

from selfcoder.simulation import run_tests_and_healthcheck
from selfcoder.self_improve import apply as apply_self_improve


ANALYSIS_DIR = Path("out/analysis_reports")


def _list_analysis_files():
    if not ANALYSIS_DIR.exists():
        return set()
    return {p for p in ANALYSIS_DIR.glob("analysis_*.json") if p.is_file()}


def test_persists_on_simulation_timeout(tmp_path):
    before = _list_analysis_files()

    # Force both pytest and healthcheck timeouts (rc=124). Even if both write
    # in the same second, at least one analysis file should (be) created/updated.
    shadow_root = Path.cwd()
    _ = run_tests_and_healthcheck(
        shadow_root,
        skip_pytest=False,
        skip_healthcheck=False,
        pytest_timeout=0,
        healthcheck_timeout=0,
    )

    after = _list_analysis_files()
    # At least one new analysis file should appear.
    assert len(after - before) >= 1


def test_persists_on_invalid_plan_json(tmp_path):
    bad = tmp_path / "bad_plan.json"
    bad.write_text("{ not: valid json\n")

    res = apply_self_improve(bad)
    assert res.get("applied") is False
    assert "analysis" in res
    assert "analysis_path" in res
    p = Path(res["analysis_path"])  # returned by apply()
    assert p.exists() and p.is_file()