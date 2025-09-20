import json
from pathlib import Path

from selfcoder.cli_ext import preflight as mod


class Args:
    def __init__(self, planfile, files=None, json=False):
        self.planfile = planfile
        self.file = files or []
        self.json = json


def test_preflight_respects_selected_files(tmp_path, monkeypatch):
    # Write a tiny plan
    plan = {"target_file": "a.py", "actions": [{"kind": "noop", "payload": {}}]}
    p = tmp_path / 'plan.json'
    p.write_text(json.dumps(plan), encoding='utf-8')

    # Monkeypatch preview to return two files
    def fake_preview(files, actions):
        # Return mapping Path->(old,new) only for requested files
        out = {}
        for f in files:
            out[Path(f)] = ("old", "new")
        return out
    monkeypatch.setattr('selfcoder.orchestrator._apply_actions_preview', fake_preview, raising=False)

    captured = {}
    def fake_review(predicted, repo_root):
        captured['files'] = sorted(list(predicted.keys()))
        return {"security": {"proceed": True, "score": 0, "findings": []}, "style": {}, "summary": {}}
    # Patch the imported symbol inside the module under test
    monkeypatch.setattr(mod, 'review_predicted_changes', fake_review, raising=False)

    # Select only b.py, ensure review sees only that file
    args = Args(str(p), files=['b.py'])
    rc = mod.cmd_preflight(args)
    assert rc == 0
    assert captured['files'] == ['b.py']
