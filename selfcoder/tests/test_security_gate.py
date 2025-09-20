
from pathlib import Path
from selfcoder.security.gate import assess_plan

def test_gate_blocks_eval(tmp_path):
    bad = 'def f():\n    return eval("1+1")\n'
    repo = Path(".").resolve()
    result = assess_plan({"bad.py": bad}, repo)
    assert result.proceed is False
    assert any(f.severity.lower() == "critical" for f in result.findings)
