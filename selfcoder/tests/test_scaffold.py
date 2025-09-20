from pathlib import Path
from selfcoder.orchestrator import apply_plan

def test_ensure_test_scaffold_idempotent(tmp_path: Path, monkeypatch):
    # Work in an isolated temp project
    monkeypatch.chdir(tmp_path)

    # Create a tiny module we’ll scaffold tests for
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    src = pkg / "m.py"
    src.write_text(
        "def greet(name: str) -> str:\n"
        "    return f'hi {name}'\n",
        encoding="utf-8",
    )

    plan = {
        "actions": [
            {"kind": "ensure_test", "payload": {"symbol": "greet", "symbol_kind": "function"}},
        ],
        "target_file": str(src),
    }

    # First apply creates tests/test_m.py with a test for greet
    apply_plan(plan, dry_run=False)
    test_file = src.parent / "tests" / "test_m.py"
    assert test_file.exists(), f"Expected scaffold at {test_file}"
    content1 = test_file.read_text(encoding="utf-8")
    assert "def test_greet" in content1

    # Second apply is idempotent (no duplicate test)
    apply_plan(plan, dry_run=False)
    content2 = test_file.read_text(encoding="utf-8")
    assert content2 == content1
