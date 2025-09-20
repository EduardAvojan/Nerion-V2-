import importlib

def test_constants_exist_and_types():
    mod = importlib.import_module('app.nerion_chat')
    # must exist
    assert hasattr(mod, 'MATCH_THRESHOLD')
    assert hasattr(mod, 'MAX_RETRIES')
    assert hasattr(mod, 'INITIAL_RETRY_DELAY')
    # basic sanity types and ranges
    assert isinstance(mod.MATCH_THRESHOLD, float)
    assert 0.0 < mod.MATCH_THRESHOLD <= 1.0
    assert isinstance(mod.MAX_RETRIES, int) and mod.MAX_RETRIES >= 1
    assert isinstance(mod.INITIAL_RETRY_DELAY, (int, float)) and mod.INITIAL_RETRY_DELAY > 0
