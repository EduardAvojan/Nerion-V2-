from __future__ import annotations

import pytest


@pytest.fixture()
def provider_files(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.yaml"
    settings = tmp_path / "settings.yaml"
    catalog.write_text(
        """
api_providers:
  defaults:
    code: openai:gpt-5
  providers:
    openai:
      endpoint: https://api.example.com/v1
      key_env: NERION_V2_OPENAI_KEY
      models:
        gpt-5:
          roles: [code]
    google:
      endpoint: https://generativelanguage.googleapis.com/v1beta
      key_env: NERION_V2_GEMINI_KEY
      models:
        gemini-2.5-pro:
          roles: [chat, code]
"""
    )
    settings.write_text(
        """
llm:
  default_provider: openai:gpt-5
  request_timeout_seconds: 5
"""
    )
    monkeypatch.setenv("NERION_MODEL_CATALOG", str(catalog))
    monkeypatch.setenv("NERION_SETTINGS_PATH", str(settings))
    return catalog, settings


def test_registry_resolves_env_override(provider_files, monkeypatch):
    from app.chat.providers import base as provider_base

    class DummyAdapter:
        def __init__(self, name: str, endpoint: str, api_key: str):
            self.name = name
            self.endpoint = endpoint
            self.api_key = api_key

        def generate(self, **_: object):  # pragma: no cover - not exercised here
            raise AssertionError("generate should not be called during resolve")

    monkeypatch.setenv("NERION_V2_OPENAI_KEY", "sk-test")
    monkeypatch.setenv("NERION_V2_CODE_PROVIDER", "openai:gpt-5")
    monkeypatch.setitem(provider_base._ADAPTERS, "openai", DummyAdapter)
    catalog, settings = provider_files
    registry = provider_base.ProviderRegistry.from_files(str(catalog), str(settings))
    adapter, model, spec = registry.resolve("code")

    assert isinstance(adapter, DummyAdapter)
    assert adapter.endpoint == "https://api.example.com/v1"
    assert adapter.api_key == "sk-test"
    assert model == "gpt-5"
    assert spec["roles"] == ["code"]

    provider_base.reset_registry()


def test_registry_uses_defaults_when_env_missing(provider_files, monkeypatch):
    from app.chat.providers import base as provider_base

    class DummyAdapter:
        def __init__(self, name: str, endpoint: str, api_key: str):
            self.name = name
            self.endpoint = endpoint
            self.api_key = api_key

        def generate(self, **_: object):  # pragma: no cover - not exercised here
            raise AssertionError("generate should not be called during resolve")

    monkeypatch.setenv("NERION_V2_OPENAI_KEY", "sk-test")
    monkeypatch.delenv("NERION_V2_CODE_PROVIDER", raising=False)
    monkeypatch.setitem(provider_base._ADAPTERS, "openai", DummyAdapter)
    provider_base.reset_registry()

    catalog, settings = provider_files
    registry = provider_base.ProviderRegistry.from_files(str(catalog), str(settings))
    adapter, model, spec = registry.resolve("code")

    assert isinstance(adapter, DummyAdapter)
    assert adapter.endpoint == "https://api.example.com/v1"
    assert adapter.api_key == "sk-test"
    assert model == "gpt-5"
    assert spec["roles"] == ["code"]

    provider_base.reset_registry()


def test_registry_resolves_google_provider(provider_files, monkeypatch):
    from app.chat.providers import base as provider_base

    class DummyAdapter:
        def __init__(self, name: str, endpoint: str, api_key: str):
            self.name = name
            self.endpoint = endpoint
            self.api_key = api_key

        def generate(self, **_: object):  # pragma: no cover - not exercised here
            raise AssertionError("generate should not be called during resolve")

    monkeypatch.setenv("NERION_V2_GEMINI_KEY", "ga-test")
    monkeypatch.setenv("NERION_V2_CODE_PROVIDER", "google:gemini-2.5-pro")
    monkeypatch.setitem(provider_base._ADAPTERS, "google", DummyAdapter)
    provider_base.reset_registry()


def test_list_role_options_and_active_defaults(provider_files, monkeypatch):
    from app.chat.providers import base as provider_base

    monkeypatch.setenv("NERION_V2_OPENAI_KEY", "sk-test")
    monkeypatch.setenv("NERION_V2_GEMINI_KEY", "ga-test")
    provider_base.reset_registry()

    catalog, settings = provider_files
    registry = provider_base.ProviderRegistry.from_files(str(catalog), str(settings))
    options = registry.list_role_options()

    assert 'code' in options
    chat_opts = options.get('chat') or []
    assert any(opt.get('provider_id') == 'google:gemini-2.5-pro' for opt in chat_opts)
    assert all(opt.get('label') for opt in chat_opts)

    assert registry.default_provider('chat') == 'openai:gpt-5'
    assert registry.active_provider('code') == 'openai:gpt-5'
    monkeypatch.setenv('NERION_V2_CODE_PROVIDER', 'google:gemini-2.5-pro')
    assert registry.active_provider('code') == 'google:gemini-2.5-pro'

    provider_base.reset_registry()
