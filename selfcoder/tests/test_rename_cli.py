import sys
import subprocess
from pathlib import Path


def test_cli_rename_applies_changes(tmp_path):
    target = tmp_path / "t.py"
    target.write_text(
        "import old.mod\nfrom old.mod import thing\nx = old.mod.thing\n",
        encoding="utf-8",
    )

    rc = subprocess.call([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod",
        "--new", "new.mod",
        str(target),
        "--apply",
    ])
    assert rc == 0

    new_src = target.read_text(encoding="utf-8")
    assert "import new.mod" in new_src
    assert "from new.mod import thing" in new_src
    assert "new.mod.thing" in new_src


def test_cli_rename_with_attr(tmp_path):
    target = tmp_path / "t.py"
    target.write_text(
        "import old.mod\nfrom old.mod import thing\nx = old.mod.thing\n",
        encoding="utf-8",
    )
    rc = subprocess.call([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod",
        "--new", "new.mod",
        "--attr-old", "thing",
        "--attr-new", "stuff",
        str(target),
        "--apply",
    ])
    assert rc == 0
    content = target.read_text(encoding="utf-8")
    assert "new.mod" in content
    assert "stuff" in content


def test_cli_rename_json_preview(tmp_path, capsys):
    import json as _json
    target_dir = tmp_path / "pkg"
    target_dir.mkdir()
    f = target_dir / "t.py"
    f.write_text(
        "import old.mod\nfrom old.mod import thing\nx = thing\n",
        encoding="utf-8",
    )

    res = subprocess.run([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--root", str(target_dir),
        "--old", "old.mod",
        "--new", "new.mod",
        "--attr-old", "thing",
        "--attr-new", "stuff",
        "--json",
    ], capture_output=True, text=True)
    assert res.returncode == 0

    out = res.stdout.strip()
    assert out.startswith("{") or out.startswith("["), f"stdout must be JSON, got: {out[:80]}"
    data = _json.loads(out)
    if isinstance(data, dict):
        files = data.get("files") or data.get("changes") or []
        assert len(files) >= 1
    elif isinstance(data, list):
        assert len(data) >= 1
    else:
        raise AssertionError("Unexpected JSON shape for --json preview")

    # preview mode should not apply changes
    content_after = f.read_text(encoding="utf-8")
    assert "old.mod" in content_after and "new.mod" not in content_after
    assert "thing" in content_after and "stuff" not in content_after


essential_cache_names = {"__pycache__"}


def test_cli_rename_root_with_include_exclude(tmp_path):
    srcdir = tmp_path / "src"
    srcdir.mkdir()

    # File that SHOULD be changed (matches include, not excluded)
    good = srcdir / "good.py"
    good.write_text(
        "import old.mod\nfrom old.mod import thing\ny = thing\n",
        encoding="utf-8",
    )

    # File that SHOULD be skipped (inside __pycache__ which is excluded)
    cache = srcdir / "__pycache__"
    cache.mkdir()
    bad = cache / "bad.py"
    bad.write_text(
        "import old.mod\nz = old.mod.thing\n",
        encoding="utf-8",
    )

    rc = subprocess.call([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--root", str(srcdir),
        "--old", "old.mod",
        "--new", "new.mod",
        "--attr-old", "thing",
        "--attr-new", "stuff",
        "--include", "*.py",
        "--exclude", "__pycache__/**",
        "--apply",
    ])
    assert rc == 0

    g = good.read_text(encoding="utf-8")
    assert "new.mod" in g and "stuff" in g
    assert "old.mod" not in g and "thing" not in g

    b = bad.read_text(encoding="utf-8")
    assert "old.mod" in b and "new.mod" not in b
    assert "thing" in b and "stuff" not in b


# Test --imports-only path
def test_cli_rename_imports_only_preview_and_apply(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    a = pkg / "a.py"
    b = pkg / "b.py"
    a.write_text("import old.mod\nfrom old.mod import x\n", encoding="utf-8")
    b.write_text("from old.mod import y\n", encoding="utf-8")

    # Dry-run preview
    res = subprocess.run([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--root", str(pkg),
        "--old", "old.mod",
        "--new", "new.mod",
        "--imports-only",
        "--json",
    ], capture_output=True, text=True)
    assert res.returncode == 0
    assert res.stdout.strip().startswith("{") or res.stdout.strip().startswith("[")

    # Apply
    rc = subprocess.call([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--root", str(pkg),
        "--old", "old.mod",
        "--new", "new.mod",
        "--imports-only",
        "--apply",
    ])
    assert rc == 0

    assert "new.mod" in a.read_text(encoding="utf-8")
    assert "new.mod" in b.read_text(encoding="utf-8")
