from __future__ import annotations

from types import SimpleNamespace
import pytest

from app.chat.providers import LLMResponse, ProviderNotConfigured
from app.parent.coder import Coder


class DummyRegistry:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(
            text=self.text,
            provider="openai",
            model="o4-mini",
            latency_s=0.1,
        )


def test_coder_uses_registry(monkeypatch):
    registry = DummyRegistry("OK")
    monkeypatch.setattr("app.parent.coder.get_registry", lambda: registry)
    coder = Coder()
    out = coder.complete("Return OK", system="sys")
    assert out == "OK"
    assert registry.calls
    call = registry.calls[0]
    assert call["role"] == "code"
    assert call["messages"][0]["role"] == "system"


def test_coder_handles_missing_provider(monkeypatch):
    class FailingRegistry:
        def generate(self, **kwargs):
            raise ProviderNotConfigured("missing")
    monkeypatch.setattr("app.parent.coder.get_registry", lambda: FailingRegistry())
    coder = Coder()
    assert coder.complete("hi") is None
