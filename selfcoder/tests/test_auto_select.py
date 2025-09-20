from __future__ import annotations

import importlib
from types import SimpleNamespace, ModuleType
import os


def test_auto_select_prefers_ollama_when_available(monkeypatch):
    # Mock Ollama tags
    class Resp:
        ok = True
        def json(self):
            return {"models": [{"name": "deepseek-coder-v2"}, {"name": "qwen2.5-coder"}]}
    import sys
    mod = ModuleType("requests")
    setattr(mod, "get", lambda url, timeout=2: Resp())
    # Ensure a minimal spec attribute so importlib.find_spec won't error later
    setattr(mod, "__spec__", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "requests", mod)

    from app.parent.selector import auto_select_model
    os.environ.pop("NERION_CODER_BASE_URL", None)
    be, model, base = auto_select_model()
    assert be == "ollama" and "coder" in model
