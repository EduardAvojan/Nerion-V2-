import json


def test_scan_plan_apply_smoke(tmp_path, monkeypatch):
    # Import module under test
    import selfcoder.self_improve as si

    # Redirect output dirs to a temp location
    reports = tmp_path / "reports"
    plans = tmp_path / "plans"
    reports.mkdir()
    plans.mkdir()
    monkeypatch.setattr(si, "REPORT_DIR", reports, raising=False)
    monkeypatch.setattr(si, "PLAN_DIR", plans, raising=False)

    # Provide a deterministic timestamp
    monkeypatch.setattr(si, "_ts", lambda: "TESTTS", raising=False)

    # Mock static analysis output → one of each tool
    def fake_run_all(_paths=None):
        return {
            "pylint": [
                {"symbol": "unused-import", "message": "unused import os", "path": "x.py", "line": 1}
            ],
            "bandit": [
                {"test_id": "B101", "issue_text": "assert used", "filename": "y.py", "line_number": 10}
            ],
            "flake8": [
                {"path": "z.py", "line": 2, "code": "F401", "text": "'sys' imported but unused"}
            ],
            "radon": [
                {"path": "w.py", "name": "f", "complexity": 14, "rank": "D", "line": 20}
            ],
        }

    monkeypatch.setattr(si, "run_all", fake_run_all, raising=False)

    # Run scan → should write a report
    rpt_path = si.scan(paths=["."])
    assert rpt_path == reports / "report_TESTTS.json"
    assert rpt_path.exists()

    # Plan from report
    pln_path = si.plan(rpt_path)
    assert pln_path == plans / "plan_TESTTS.json"
    data = json.loads(pln_path.read_text())
    assert "actions" in data and isinstance(data["actions"], list)
    assert any(a.get("action") == "remove_unused_imports" for a in data["actions"])  # from pylint/flake8
    assert any(a.get("action") == "security_refactor" for a in data["actions"])     # from bandit
    assert any(a.get("action") == "extract_function" for a in data["actions"])      # from radon

    # Mock apply stack to avoid touching git or running real tests
    monkeypatch.setattr(si, "snapshot", lambda: "SNAP", raising=False)
    monkeypatch.setattr(si, "restore_snapshot", lambda **kwargs: (0, None), raising=False)

    class DummySim:
        @staticmethod
        def apply(plan_json):
            # sanity: plan should be the one we generated
            assert plan_json.get("version") == 1
    monkeypatch.setattr(si, "SimulationMode", DummySim, raising=False)

    class DummyOrch:
        @staticmethod
        def apply_plan(plan_json):
            pass
    monkeypatch.setattr(si, "Orchestrator", DummyOrch, raising=False)

    monkeypatch.setattr(si, "run_healthcheck", lambda: True, raising=False)

    res = si.apply(pln_path, simulate=True)
    assert res.get("applied") is True
    assert res.get("rolled_back") is False
    assert res.get("snapshot") == "SNAP"