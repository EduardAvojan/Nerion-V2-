import importlib


def test_fuzzy_matches_command_is_callable():
    mod = importlib.import_module("app.nerion_chat")
    assert hasattr(mod, "fuzzy_matches_command"), (
        "Function fuzzy_matches_command must exist in app.nerion_chat"
    )
    fn = getattr(mod, "fuzzy_matches_command")
    assert callable(fn)
