
from pathlib import Path
import tempfile

def test_iter_pyfiles_skips_noise_and_finds_code(tmp_path):
    root = tmp_path
    (root / "keep_a.py").write_text("a=1\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "keep_b.py").write_text("b=2\n", encoding="utf-8")

    (root / ".venv").mkdir()
    (root / ".venv" / "bad.py").write_text("x=0\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "c.py").write_text("x=0\n", encoding="utf-8")

    from selfcoder.actions.crossfile import _iter_pyfiles
    files = {p.name for p in _iter_pyfiles(root)}
    assert "keep_a.py" in files
    assert "keep_b.py" in files
    assert "bad.py" not in files
    assert "c.py" not in files
