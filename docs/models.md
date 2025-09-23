# Nerion V2 Model Configuration

Nerion V2 uses hosted LLM providers configured via `config/model_catalog.yaml` and `app/settings.yaml`.

## Provider Catalog (`config/model_catalog.yaml`)
- `api_providers.defaults` — per-role fallbacks (chat, code, planner, embeddings).
- `api_providers.providers` — provider entries with:
  - `endpoint` (base URL for the API)
  - `key_env` (environment variable that must contain the API key)
  - `models` → nested mapping of `model_name` to metadata (roles, limits, capabilities).

Example:
```yaml
api_providers:
  defaults:
    chat: openai:gpt-5
  providers:
    openai:
      endpoint: https://api.openai.com/v1
      key_env: NERION_V2_OPENAI_KEY
      models:
        gpt-5:
          roles: [chat, code]
          max_output_tokens: 30000
    google:
      endpoint: https://generativelanguage.googleapis.com/v1beta
      key_env: NERION_V2_GEMINI_KEY
      models:
        gemini-2.5-pro:
          roles: [chat, code, planner]
          supports_structured_output: true
```

## Runtime Defaults (`app/settings.yaml`)
- `llm.default_provider` and `llm.fallback_provider` set the primary/fallback provider IDs.
- `llm.roles` (optional) overrides per-role providers.
- `llm.request_timeout_seconds` and `llm.max_cost_per_turn_usd` guard runtime usage.

## Environment Overrides
- Copy `.env.example` to `.env` and populate provider keys.
- Override defaults per shell/session:
  - `NERION_V2_DEFAULT_PROVIDER`
  - `NERION_V2_CHAT_PROVIDER`
  - `NERION_V2_CODE_PROVIDER`
  - `NERION_V2_PLANNER_PROVIDER`
  - `NERION_V2_EMBEDDINGS_PROVIDER`
  - `NERION_V2_REQUEST_TIMEOUT`

Providers are expressed as `provider:model` identifiers (for example `openai:gpt-5`).

Older offline backends (Ollama, llama.cpp, vLLM, exllamav2) are no longer part of the V2 stack. If you need the
previous local-first experience, use the classic Nerion repository instead.
