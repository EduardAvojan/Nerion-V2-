import importlib


def _get_fn():
    mod = importlib.import_module("app.nerion_chat")
    assert hasattr(mod, "matches_command"), (
        "Function matches_command must exist in app.nerion_chat"
    )
    fn = getattr(mod, "matches_command")
    assert callable(fn)
    return (mod, fn)


def test_matches_command_exact_true():
    (mod, fn) = _get_fn()
    phrases = {"mute", "unmute", "sleep"}
    assert fn("mute", phrases) is True
    assert fn("unmute", phrases) is True


def test_matches_command_clear_false():
    (mod, fn) = _get_fn()
    phrases = {"mute", "unmute", "sleep"}
    assert fn("volume up", phrases) is False
    assert fn("open browser", phrases) is False


def test_matches_command_avoid_substring_collision():
    (mod, fn) = _get_fn()
    phrases = {"mute", "unmute", "sleep"}
    assert fn("mute please", phrases) is True
    assert fn("please unmute", phrases) is True
    assert fn("mutes", phrases) is False
    assert fn("unmute now", phrases) is True
    assert fn("mute", phrases) is True
    assert fn("unmute", phrases) is True
