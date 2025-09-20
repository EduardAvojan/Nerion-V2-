import os
import tempfile
import yaml

import app.chat.engine as engine


def test_load_tools_manifest_from_yaml(tmp_path):
    # Create a fake config/tools.yaml with one tool
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    yaml_path = cfg_dir / "tools.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "tools": [
                    {
                        "name": "dummy_tool",
                        "description": "A fake test tool",
                        "requires_network": False,
                        "params": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Temporarily chdir so engine looks in this temp config folder
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        manifest = engine._load_tools_manifest()
        assert manifest.tools, "Expected at least one tool in manifest"
        assert manifest.tools[0].name == "dummy_tool"
    finally:
        os.chdir(old_cwd)