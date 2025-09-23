from __future__ import annotations

from pathlib import Path

from selfcoder.security.gate import assess_plan


def test_default_allowlist_blocks_unknown_directory(tmp_path: Path):
    repo_root = tmp_path
    (repo_root / 'config').mkdir()
    # Copy default policy file to repo root config
    from shutil import copyfile
    copyfile(Path('config/self_mod_policy.yaml'), repo_root / 'config' / 'self_mod_policy.yaml')

    plan_changes = {str(repo_root / 'forbidden' / 'payload.bin'): 'binary'}
    res = assess_plan(plan_changes, repo_root)
    assert res.proceed is False
    assert any(f.rule_id == 'POLICY:PATH' for f in res.findings)


def test_default_allowlist_permits_known_directory(tmp_path: Path):
    repo_root = tmp_path
    (repo_root / 'config').mkdir()
    from shutil import copyfile
    copyfile(Path('config/self_mod_policy.yaml'), repo_root / 'config' / 'self_mod_policy.yaml')

    allowed_file = repo_root / 'app' / 'main.py'
    allowed_file.parent.mkdir(parents=True, exist_ok=True)
    plan_changes = {str(allowed_file): 'print("ok")'}
    res = assess_plan(plan_changes, repo_root)
    assert res.proceed is True
