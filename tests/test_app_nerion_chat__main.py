import importlib


def test_main_is_callable():
    mod = importlib.import_module('app.nerion_chat')
    assert hasattr(mod, 'main'), 'Function main must exist in app.nerion_chat'
    fn = getattr(mod, 'main')
    assert callable(fn)
