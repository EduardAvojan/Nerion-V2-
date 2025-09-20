from __future__ import annotations

import json
from pathlib import Path

import selfcoder.cli_ext.preflight as pf


def test_preflight_cli_json(monkeypatch, tmp_path):
    # Create a small plan file
    target = Path("tmp_preflight.py")
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    plan = {
        "target_file": str(target),
        "actions": [
            {"kind": "add_module_docstring", "payload": {"doc": "P"}},
        ],
    }
    planfile = tmp_path / "plan.json"
    planfile.write_text(json.dumps(plan), encoding="utf-8")

    # Stub preview to avoid running transforms
    fake_preview = {target: ("old", '"""P"""\n' + target.read_text(encoding='utf-8'))}
    monkeypatch.setattr("selfcoder.orchestrator._apply_actions_preview", lambda files, actions: fake_preview, raising=False)

    # Run preflight
    ns = type("Args", (), {"planfile": str(planfile), "json": True})
    rc = pf.cmd_preflight(ns)
    assert rc == 0

