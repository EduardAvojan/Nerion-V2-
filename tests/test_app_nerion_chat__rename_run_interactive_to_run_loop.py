import importlib


def test_rename_run_interactive_to_run_loop():
    mod = importlib.import_module("app.nerion_chat")
    assert not hasattr(mod, "run_interactive"), (
        "Old symbol run_interactive should be gone after rename"
    )
    assert hasattr(mod, "run_loop"), "New symbol run_loop must exist after rename"
    assert callable(getattr(mod, "run_loop"))
