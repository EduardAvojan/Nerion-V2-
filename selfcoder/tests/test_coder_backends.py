from __future__ import annotations

import importlib
from types import SimpleNamespace
import os


def test_ollama_backend_selected(monkeypatch):
    class DummyChat:
        def __init__(self, **k):
            pass
        def invoke(self, messages):
            return SimpleNamespace(content="OK")
    import sys
    sys.modules["langchain_ollama"] = SimpleNamespace(ChatOllama=DummyChat)
    os.environ["NERION_CODER_BACKEND"] = "ollama"
    os.environ["NERION_CODER_MODEL"] = "deepseek-coder-v2"
    from app.parent.coder import Coder
    c = Coder()
    out = c.complete("Reply with OK only.")
    assert out and "OK" in out
    os.environ.pop("NERION_CODER_BACKEND", None)
    os.environ.pop("NERION_CODER_MODEL", None)


def test_llama_cpp_unavailable_returns_none(monkeypatch):
    # Ensure import fails
    import sys
    sys.modules.pop("llama_cpp", None)
    os.environ["NERION_CODER_BACKEND"] = "llama_cpp"
    from app.parent.coder import Coder
    c = Coder(model="codellama")
    out = c.complete("hi")
    assert out is None
    os.environ.pop("NERION_CODER_BACKEND", None)
