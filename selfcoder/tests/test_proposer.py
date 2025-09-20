from __future__ import annotations

import json
from pathlib import Path

from selfcoder.repair import proposer as P


def _ctx_for(tmp: Path, failures):
    return {"_task_dir": str(tmp.resolve()), "failures": failures}


def test_proposer_adds_missing_module_import(tmp_path: Path):
    # Create a source file that uses a missing module "xfoo"
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    src = pkg / "mod.py"
    src.write_text("""
def run():
    return xfoo.do()
""".lstrip(), encoding="utf-8")
    # Simulate failure record with ModuleNotFoundError and top frame in mod.py
    failures = [
        {
            "traceback": "ModuleNotFoundError: No module named 'xfoo'",
            "frames": [{"file": str(src)}],
        }
    ]
    ctx = _ctx_for(tmp_path, failures)
    diffs = P.propose_diff_multi(ctx)
    assert diffs and isinstance(diffs[0], str)
    d = diffs[0]
    assert "+++ b/pkg/mod.py" in d and "--- a/pkg/mod.py" in d
    assert "import xfoo" in d


def test_proposer_adds_missing_module_import_importerror(tmp_path: Path):
    # Same as above but with ImportError wording (alternative pattern)
    pkg = tmp_path / "pkg2"
    pkg.mkdir()
    src = pkg / "mod2.py"
    src.write_text("""
def run():
    return ylib.do()
""".lstrip(), encoding="utf-8")
    failures = [
        {
            "traceback": "ImportError: No module named ylib",
            "frames": [{"file": str(src)}],
        }
    ]
    ctx = _ctx_for(tmp_path, failures)
    diffs = P.propose_diff_multi(ctx)
    assert diffs and any("import ylib" in d for d in diffs)


def test_proposer_uses_grep_when_no_frames(tmp_path: Path):
    # Repo has two modules; only one references token 'xfoo'
    (tmp_path / 'a.py').write_text('def a():\n    return 0\n', encoding='utf-8')
    src = tmp_path / 'b.py'
    src.write_text('def run():\n    return xfoo.do()\n', encoding='utf-8')
    # No frames in failure; message hints at missing module 'xfoo'
    failures = [{"traceback": "ModuleNotFoundError: No module named 'xfoo'"}]
    ctx = {"_task_dir": str(tmp_path.resolve()), "failures": failures}
    diffs = P.propose_diff_multi(ctx)
    assert diffs and "b.py" in diffs[0]


def test_proposer_alias_np_import(tmp_path: Path):
    # File using np without import
    f = tmp_path / "a.py"
    f.write_text("""
def f(x):
    return np.array(x)
""".lstrip(), encoding="utf-8")
    failures = [
        {
            "traceback": "NameError: name 'np' is not defined",
            "frames": [{"file": str(f)}],
        }
    ]
    ctx = _ctx_for(tmp_path, failures)
    diffs = P.propose_diff_multi(ctx)
    assert any("import numpy as np" in d for d in diffs)
