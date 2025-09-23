from __future__ import annotations

import json
from pathlib import Path
import argparse

import pytest

from selfcoder.planner import architect_briefs


@pytest.fixture()
def tmp_reports(tmp_path: Path, monkeypatch):
    reports_dir = tmp_path / "analysis"
    reports_dir.mkdir(parents=True)
    monkeypatch.setattr(architect_briefs, "REPORTS_DIR", reports_dir)
    return reports_dir


@pytest.fixture()
def tmp_roadmap(tmp_path: Path, monkeypatch):
    roadmap = tmp_path / "AGENTS.md"
    roadmap.write_text(
        """
### Phase 3 – Autonomous Planning
- [ ] Architect Brief Generator
- [x] Policy-aware prioritiser
- [ ] Planner integration
### Phase 4 – Execution & Verification
- [ ] Simulation harness
""".strip()
    )
    monkeypatch.setattr(architect_briefs, "ROADMAP_FILE", roadmap)
    return roadmap


@pytest.fixture()
def coverage_file(tmp_path: Path, monkeypatch):
    cov = tmp_path / "coverage.json"
    payload = {
        "totals": {"covered_lines": 80, "num_statements": 100},
        "files": {
            "core/module.py": {
                "summary": {"missing_lines": 12},
            }
        },
    }
    cov.write_text(json.dumps(payload))
    monkeypatch.setattr(architect_briefs, "COVERAGE_FILE", cov)
    monkeypatch.setattr(architect_briefs, "load_baseline", lambda: None)
    return cov


def test_generate_architect_briefs_combines_signals(tmp_reports, tmp_roadmap, coverage_file, monkeypatch):
    report = tmp_reports / "report_1.json"
    report.write_text(
        json.dumps(
            {
                "pylint": [
                    {
                        "symbol": "unused-import",
                        "message": "unused import",
                        "path": "core/module.py",
                    }
                ]
            }
        )
    )

    snapshot = {
        "knowledge_graph": {
            "hotspots": [
                {
                    "component": "core",
                    "risk_score": 4.5,
                    "test_failures": 2,
                    "apply_failures": 1,
                    "recent_fix_commits": 3,
                }
            ]
        },
        "anomalies": ["Hotspot core failing apply"],
    }

    monkeypatch.setattr(architect_briefs, "load_operator_snapshot", lambda window_hours=48: snapshot)

    briefs = architect_briefs.generate_architect_briefs(max_briefs=3)
    assert briefs
    first = briefs[0]
    assert first.component == "core"
    assert any("Telemetry shows" in line for line in first.rationale)
    assert any("Coverage report" in line for line in first.rationale)
    assert any("telemetry" in line.lower() or "coverage" in line.lower() for line in first.acceptance_criteria)
    assert first.priority > 0


def test_architect_cli_json(monkeypatch, tmp_reports, tmp_roadmap, coverage_file, capsys):
    report = tmp_reports / "report_2.json"
    report.write_text(json.dumps({"pylint": []}))

    snapshot = {
        "knowledge_graph": {
            "hotspots": [
                {
                    "component": "app",
                    "risk_score": 3.0,
                    "test_failures": 0,
                    "apply_failures": 0,
                    "recent_fix_commits": 1,
                }
            ]
        }
    }
    monkeypatch.setattr(architect_briefs, "load_operator_snapshot", lambda window_hours=48: snapshot)

    from selfcoder.cli_ext import architect as architect_cli

    ns = argparse.Namespace(max=2, window=24, json=True, no_smells=True)
    rc = architect_cli._cmd_briefs(ns)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data[0]["component"] == "app"
