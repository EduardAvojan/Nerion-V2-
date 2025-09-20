# selfcoder/tests/test_coverage_integration.py
from __future__ import annotations
from pathlib import Path
import pytest
import json
from selfcoder import coverage_utils

def test_coverage_baseline_roundtrip(tmp_path: Path):
    """Ensure baseline can be saved and loaded correctly using the actual data format."""
    baseline_file = tmp_path / "baseline.json"
    
    # Create a dummy coverage report data structure
    dummy_report = {
        "totals": {
            "covered_lines": 71,
            "num_statements": 100,
            "percent_covered": 71.0
        }
    }

    # Use the functions from your file
    coverage_utils.save_baseline(dummy_report, path=baseline_file)
    loaded_data = coverage_utils.load_baseline(path=baseline_file)

    assert baseline_file.exists()
    assert isinstance(loaded_data, dict)
    
    # Check the actual value using your overall_percent function
    percent = coverage_utils.overall_percent(loaded_data)
    assert abs(percent - 71.0) < 1e-6

def test_healthcheck_fails_on_coverage_drop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Simulate a health check failing when coverage drops."""
    baseline_file = tmp_path / "baseline.json"
    
    # 1. Create a high baseline report and save it
    high_coverage_report = {"totals": {"covered_lines": 85, "num_statements": 100}}
    coverage_utils.save_baseline(high_coverage_report, path=baseline_file)

    # 2. Mock the function that runs pytest to return a lower coverage report
    low_coverage_report = {"totals": {"covered_lines": 80, "num_statements": 100}}
    monkeypatch.setattr(coverage_utils, "run_pytest_with_coverage", lambda *args, **kwargs: low_coverage_report)

    # 3. Simulate the health check logic
    baseline = coverage_utils.load_baseline(path=baseline_file)
    current = coverage_utils.run_pytest_with_coverage(pytest_args=[]) # Args don't matter due to mock
    
    current_pct, delta = coverage_utils.compare_to_baseline(current, baseline)

    # 4. Assert that the drop is detected
    assert current_pct == 80.0
    assert delta < 0
    assert abs(delta - (-5.0)) < 1e-6