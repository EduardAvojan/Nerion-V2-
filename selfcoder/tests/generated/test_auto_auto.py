import pytest
from pathlib import Path

def test_generated_plan_applied():
    src = Path(r"""/tmp/auto.py""").read_text(encoding="utf-8")
    assert '"""AutoTest Guard"""' in src or "'''AutoTest Guard'''" in src
from selfcoder.cli import main

def test_generated_plan_applied(tmp_path):
    # Create a temp module to operate on
    target = tmp_path / "auto.py"
    target.write_text(
        "def ping():\n    return 'pong'\n",
        encoding="utf-8",
    )

    # Use CLI to apply a simple plan
    rc = main([
        "plan",
        "-i", "add module docstring 'Auto demo'",
        "-f", str(target),
        "--apply",
        "--skip-healthcheck",
        "--skip-pytest",
    ])
    assert rc == 0

    # Validate the change landed
    src = target.read_text(encoding="utf-8")
    assert src.lstrip().startswith('"""')
    assert "Auto demo" in src