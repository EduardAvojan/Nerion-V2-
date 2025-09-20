import pytest
from pathlib import Path
from textwrap import dedent

from selfcoder.orchestrator import run_batch_actions_on_files, apply_plan

def test_batch_inserts_docstrings(tmp_path):
    file1 = tmp_path / "file1.py"
    file2 = tmp_path / "file2.py"
    content = "def foo():\n    pass\n"
    file1.write_text(content)
    file2.write_text(content)

    actions = [
        {"action": "add_module_docstring", "payload": {"docstring": "Module docstring"}}
    ]

    run_batch_actions_on_files([file1, file2], actions, dry_run=False)

    result1 = file1.read_text()
    result2 = file2.read_text()

    assert '"""Module docstring"""' in result1
    assert '"""Module docstring"""' in result2

def test_batch_respects_dry_run(tmp_path):
    file = tmp_path / "file.py"
    original_content = "def foo():\n    pass\n"
    file.write_text(original_content)

    actions = [
        {"action": "add_module_docstring", "payload": {"docstring": "Module docstring"}}
    ]

    run_batch_actions_on_files([file], actions, dry_run=True)

    result = file.read_text()
    assert result == original_content


def test_apply_rolls_back_on_unresolved_imports(tmp_path):
    # Prepare a file that has an unresolved import; any touch should trigger postcondition failure
    target = tmp_path / "bad_import.py"
    original = "import definitely_not_a_real_module\n\n"  # keep unparsable import name
    target.write_text(original)

    # Build a minimal plan that would modify the file (so rollback has an effect)
    plan = {
        "actions": [
            {
                "kind": "insert_function",
                "payload": {
                    "name": "helper",
                    "content": "def helper():\n    return 42\n",
                },
            }
        ],
        "target_file": str(target),
        # This postcondition should fail due to the unresolved import above
        "postconditions": ["no_unresolved_imports"],
    }

    touched = apply_plan(plan, dry_run=False)

    # Orchestrator should detect failure and roll back AST edits
    assert touched == []
    assert target.read_text(encoding="utf-8") == original
