from core.ui.prefs import load_prefs, save_prefs
def test_prefs_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = {"overlay":{"filter":"eslint"}}
    save_prefs(d)
    got = load_prefs()
    assert got["overlay"]["filter"] == "eslint"
