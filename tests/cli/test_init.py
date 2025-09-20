import os, pathlib, shutil
from selfcoder.cli_init import main

def test_init_writes(tmp_path, monkeypatch):
    cwd = tmp_path
    monkeypatch.chdir(cwd)
    rc = main([])
    assert rc == 0
    assert (cwd/".nerion/policy.yaml").exists()
    assert (cwd/"app/settings.yaml").exists()
