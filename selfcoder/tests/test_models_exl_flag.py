from __future__ import annotations

import os


def test_coder_exllamav2_graceful_when_unavailable(monkeypatch):
    # Force exllamav2 backend with a dummy model dir; expect graceful None output
    monkeypatch.setenv("NERION_CODER_BACKEND", "exllamav2")
    monkeypatch.setenv("EXLLAMA_MODEL_DIR", "/nonexistent/path")
    from app.parent.coder import Coder
    c = Coder(model="codellama-7b-instruct-gptq", backend="exllamav2")
    out = c.complete("Respond with OK")
    assert out is None

