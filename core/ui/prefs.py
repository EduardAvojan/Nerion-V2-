import json
import os
import pathlib

def _prefs_path() -> pathlib.Path:
    """Resolve prefs path at call time to honor HOME changes in tests."""
    # Allow override via NERION_UI_PREFS for flexibility
    override = os.getenv("NERION_UI_PREFS")
    if override:
        return pathlib.Path(override)
    # Compute under HOME each time so monkeypatching HOME works
    home = os.path.expanduser("~")
    return pathlib.Path(home) / ".nerion" / "ui.json"

def load_prefs() -> dict:
    p = _prefs_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_prefs(d: dict) -> None:
    p = _prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
