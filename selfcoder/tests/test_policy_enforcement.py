from __future__ import annotations

from pathlib import Path
from selfcoder.orchestrator import apply_plan
from selfcoder.security.gate import assess_plan


def test_policy_blocks_denied_action(tmp_path: Path, monkeypatch):
    # Create a minimal repo with config/policy.yaml that denies rename_symbol
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'policy.yaml').write_text('actions:\n  deny: [rename_symbol]\n', encoding='utf-8')
    # A target file to reference
    f = tmp_path / 'a.py'
    f.write_text('def x():\n    return 1\n', encoding='utf-8')
    # Plan that attempts a denied action
    plan = {
        'actions': [{'kind': 'rename_symbol', 'payload': {'from': 'x', 'to': 'y'}}],
        'target_file': str(f),
    }
    # Run in the temp repo cwd
    import os
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        out = apply_plan(plan, dry_run=False)
        assert out == []
    finally:
        os.chdir(cwd)


def test_policy_blocks_denied_path(tmp_path: Path):
    # Deny plugins/** path via policy
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'policy.yaml').write_text('paths:\n  deny: ["plugins/**"]\n', encoding='utf-8')
    # Predicted change to a denied path
    res = assess_plan({str(tmp_path / 'plugins' / 'bad.py'): 'print(1)\n'}, tmp_path)
    assert res.proceed is False
    assert any('POLICY:PATH' in f.rule_id for f in res.findings)

