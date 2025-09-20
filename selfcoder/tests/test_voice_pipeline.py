import importlib
import pytest

pytestmark = pytest.mark.voice

@pytest.mark.skipif(
    not hasattr(importlib.util.find_spec("app.nerion_chat"), "loader"),
    reason="app.nerion_chat module not available"
)
def test_voice_pipeline_applies_docstring(tmp_path):
    import importlib
    nc = importlib.import_module("app.nerion_chat")

    if not hasattr(nc, "run_self_coding_pipeline"):
        pytest.skip("run_self_coding_pipeline not available in app.nerion_chat")

    # Create a target file OUTSIDE the repo (tmp_path is fine)
    target = tmp_path / "voice_target.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")

    # Stub out I/O: no speaking, and "hear" the path to our file
    nc.speak = lambda *a, **k: None
    nc.listen_once = lambda **k: str(target)

    # Run the pipeline with a simple instruction
    ok = nc.run_self_coding_pipeline("add module docstring 'Voice Test'")
    assert ok is True

    # Verify the file received the docstring
    new_src = target.read_text(encoding="utf-8")
    assert '"""Voice Test"""' in new_src or "'''Voice Test'''" in new_src
