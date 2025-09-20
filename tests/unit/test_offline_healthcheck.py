import sys
import types

def test_offline_healthcheck_formats_bool(monkeypatch):
    fake_mod = types.SimpleNamespace()
    fake_mod.run_all = lambda verbose=False: True
    monkeypatch.setitem(sys.modules, 'selfcoder', types.SimpleNamespace(healthcheck=fake_mod))
    monkeypatch.setitem(sys.modules, 'selfcoder.healthcheck', fake_mod)
    from app.chat.offline_tools import run_healthcheck
    out = run_healthcheck(None)
    assert "Healthcheck: " in out

