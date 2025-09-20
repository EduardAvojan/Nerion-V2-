from __future__ import annotations

import os
from pathlib import Path

import selfcoder.orchestrator as orch


def test_test_impact_with_tester_expansion(monkeypatch, tmp_path):
    # Create a target file inside repo root
    target = Path("tmp_impact_target.py")
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")

    plan = {
        "target_file": str(target),
        "actions": [
            {"kind": "add_module_docstring", "payload": {"doc": "ImpactDoc"}},
        ],
    }

    # Enable impact + tester via env
    monkeypatch.setenv("NERION_TEST_IMPACT", "1")
    monkeypatch.setenv("NERION_TESTER", "1")

    # Force no initially impacted tests to trigger scaffolding path
    monkeypatch.setattr(orch, "_predict_impacted_tests", lambda *_a, **_k: [], raising=False)

    # Capture writes and pytest runs
    writes = []
    runs = []

    def _fake_write(code: str, out_path):
        writes.append(str(out_path))
        return Path(out_path)

    def _fake_run(paths):
        runs.extend([str(p) for p in paths])
        return 0

    monkeypatch.setattr(orch._testgen, "write_test_file", _fake_write, raising=False)
    monkeypatch.setattr(orch._testgen, "run_pytest_on_paths", _fake_run, raising=False)

    # Apply the plan (in-repo path); should not raise
    _ = orch.apply_plan(plan, dry_run=False, preview=False)

    # Ensure an edge or auto test was generated and scheduled to run
    assert any("test_edge_" in p or "test_auto_" in p for p in writes)
    assert any("test_edge_" in p or "test_auto_" in p for p in runs)

