import importlib


def test_with_retries_is_callable():
    mod = importlib.import_module('app.nerion_chat')
    assert hasattr(mod, 'with_retries'), 'Function with_retries must exist in app.nerion_chat'
    fn = getattr(mod, 'with_retries')
    assert callable(fn)
