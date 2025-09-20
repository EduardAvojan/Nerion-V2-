import importlib


def test_run_loop_is_callable():
    mod = importlib.import_module('app.nerion_chat')
    assert hasattr(mod, 'run_loop'), 'Function run_loop must exist in app.nerion_chat'
    fn = getattr(mod, 'run_loop')
    assert callable(fn)
