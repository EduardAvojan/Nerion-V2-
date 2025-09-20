
import subprocess, sys, textwrap
from pathlib import Path

def _write(path: Path, text: str):
    path.write_text(textwrap.dedent(text), encoding="utf-8")

def test_scan_fail_on_critical(tmp_path):
    target = tmp_path / "evil.py"
    _write(target, """
    def f():
        return eval("1+1")
    """)
    rc_ok = subprocess.call([sys.executable, "-m", "selfcoder.cli", "scan", str(target)])
    assert rc_ok == 0
    rc_fail = subprocess.call([sys.executable, "-m", "selfcoder.cli", "scan", str(target), "--fail-on", "critical"])
    assert rc_fail != 0
