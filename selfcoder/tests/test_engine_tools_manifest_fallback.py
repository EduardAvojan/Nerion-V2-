import os
import app.chat.engine as engine

def test_load_tools_manifest_fallback(tmp_path):
    # Make sure no config/tools.yaml exists
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    yaml_path = cfg_dir / "tools.yaml"
    if yaml_path.exists():
        yaml_path.unlink()

    # Temporarily chdir so engine looks in this temp config folder
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        manifest = engine._load_tools_manifest()
        assert manifest is not None
        assert hasattr(manifest, "tools")
        # Should return an empty manifest safely
        assert manifest.tools == []
    finally:
        os.chdir(old_cwd)