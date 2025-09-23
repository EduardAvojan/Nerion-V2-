from __future__ import annotations

import os


def test_auto_select_reads_provider_env(monkeypatch):
    from app.parent.selector import auto_select_model

    monkeypatch.setenv("NERION_V2_CODE_PROVIDER", "openai:gpt-5")
    assert auto_select_model() == ("openai", "gpt-5", None)
    monkeypatch.delenv("NERION_V2_CODE_PROVIDER", raising=False)
    assert auto_select_model() is None
