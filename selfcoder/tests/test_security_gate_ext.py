from __future__ import annotations

from pathlib import Path
from selfcoder.security.gate import assess_plan


def test_gate_blocks_on_critical_eval(tmp_path: Path):
    # Using eval() should produce a critical finding via AST scanner
    code = "def f():\n    return eval('1')\n"
    res = assess_plan({str(tmp_path / 'a.py'): code}, tmp_path)
    assert res.proceed is False
    assert 'critical' in res.reason or res.score >= 10


def test_gate_allows_high_when_under_threshold(tmp_path: Path, monkeypatch):
    # os.system('ls') is high severity; with balanced threshold (10) it should not block by score alone
    code = "import os\n\n\n    def f():\n        return os.system('ls')\n"
    # Ensure policy is balanced
    monkeypatch.setenv('NERION_POLICY', 'balanced')
    res = assess_plan({str(tmp_path / 'b.py'): code}, tmp_path)
    assert res.proceed is True
