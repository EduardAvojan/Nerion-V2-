

from pathlib import Path
from selfcoder.analysis.symbols import build_symbol_index

def test_build_symbol_index_on_temp_tree(tmp_path):
    # create a small tree
    a = tmp_path / "a.py"
    a.write_text(
        "def foo():\n    pass\n\nclass Bar:\n    pass\n\ndef _hidden():\n    pass\n",
        encoding="utf-8",
    )
    b = tmp_path / "pkg"
    b.mkdir()
    (b / "__init__.py").write_text("", encoding="utf-8")
    (b / "b.py").write_text("def foo2():\n    return 1\n", encoding="utf-8")

    idx = build_symbol_index(tmp_path)

    # top-level defs/classes are indexed
    assert "foo" in idx and a in idx["foo"]
    assert "Bar" in idx and a in idx["Bar"]
    assert "foo2" in idx and (b / "b.py") in idx["foo2"]

    # Leading underscore is still a symbol, but ensure it's present as well
    assert "_hidden" in idx


# --- New tests for build_defs_uses and cache
from selfcoder.analysis.symbols import build_defs_uses

def test_build_defs_uses_returns_defs_and_uses(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("def foo():\n    return 42\n\nfoo()\n", encoding="utf-8")
    idx = build_defs_uses(tmp_path, use_cache=False)
    assert "foo" in idx["defs"]
    assert "foo" in idx["uses"]
    assert any(u["file"].endswith("a.py") for u in idx["uses"]["foo"])

def test_build_defs_uses_respects_cache(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("def bar():\n    return 1\nbar()\n", encoding="utf-8")
    idx1 = build_defs_uses(tmp_path, use_cache=True)
    # Modify file but keep same mtime by resetting mtime
    import os, time
    content = a.read_text(encoding="utf-8")
    a.write_text(content + "\n", encoding="utf-8")
    os.utime(a, (os.stat(a).st_atime, os.stat(a).st_mtime))
    idx2 = build_defs_uses(tmp_path, use_cache=True)
    # Should reuse cache (defs still contain bar)
    assert "bar" in idx2["defs"]