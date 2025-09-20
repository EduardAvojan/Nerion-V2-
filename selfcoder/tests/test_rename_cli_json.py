# selfcoder/tests/test_rename_cli_json.py
import subprocess
import sys
import json

def test_cli_rename_json_preview(tmp_path):
    """
    Ensure the `nerion rename --json` command outputs a valid JSON object
    that contains the file path and its new content.
    """
    srcdir = tmp_path / "src"
    srcdir.mkdir()
    
    file_to_change = srcdir / "app.py"
    file_to_change.write_text("import old_api\nresult = old_api.call()\n")

    # Run the command and capture stdout
    result = subprocess.run([
        sys.executable, "-m", "selfcoder.cli", "rename",
        "--old", "old_api",
        "--new", "new_api",
        "--json",
        str(file_to_change)
    ], capture_output=True, text=True, check=True)

    # Parse the JSON output
    output_data = json.loads(result.stdout)

    # Assertions
    expected_path = str(file_to_change)
    expected_content = "import new_api\nresult = new_api.call()\n"

    # --- FIX IS HERE ---
    # The test now checks the nested structure of the JSON
    assert isinstance(output_data, dict)
    assert "files" in output_data
    assert isinstance(output_data["files"], list)
    assert len(output_data["files"]) == 1

    file_info = output_data["files"][0]
    assert file_info.get("path") == expected_path
    assert file_info.get("content") == expected_content