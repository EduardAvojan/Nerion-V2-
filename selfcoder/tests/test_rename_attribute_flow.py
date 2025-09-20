# selfcoder/tests/test_rename_attribute_flow.py
import subprocess
import sys

def test_attribute_rename_handles_both_cases(tmp_path):
    """
    A smoke test to ensure that --attr-old/--attr-new correctly renames an attribute
    in both `from ... import ...` statements and in `module.attribute` dotted access.
    """
    test_file = tmp_path / "app.py"
    test_file.write_text(
        "from old.mod import old_thing\n"
        "result = old.mod.old_thing()\n"
    )

    # Run the rename command
    subprocess.run([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old.mod",
        "--new", "new.mod",
        "--attr-old", "old_thing",
        "--attr-new", "new_thing",
        "--apply",
        str(test_file)
    ], check=True)

    # Read the modified content
    content = test_file.read_text()

    # Define what the content should look like after the rename
    expected_content = (
        "from new.mod import new_thing\n"
        "result = new.mod.new_thing()\n"
    )

    # Assert that the content is exactly as expected
    assert content == expected_content
