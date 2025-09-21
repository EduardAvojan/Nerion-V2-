# Nerion V2 – API LLM Architecture

## Goals
- Deliver a responsive, reliable assistant by leaning on hosted LLM APIs instead of on-device checkpoints.
- Preserve Nerion differentiators (self-coding, planner tooling, HOLO UI) while redefining "privacy-first" as "user data never leaves the device without consent".
- Provide a clean migration path for existing Nerion users and a forward-looking contract for future integrations.

## Target Providers
| Priority | Provider | Use Case | Notes |
| --- | --- | --- | --- |
| P0 | OpenAI (o4-mini / gpt-4o) | general chat + coder | strong tooling ecosystem, streaming support |
| P0 | Anthropic (Claude 3.5 Sonnet) | safety-critical planner replies | longer context, reliable structured output |
| P1 | Google (Gemini 2.0 Flash) | fallback + multimodal | fast latency, vision input |
| P2 | Local gateway (optional) | privacy conscious deployments | shim for self-hosted vLLM or Ollama

Each provider registers through a thin adapter layer (`app/chat/providers` in Phase 2) that exposes:
- `async generate(messages, *, mode, metadata) -> Stream[Chunk]`
- `supports_tool_calls`
- `max_tokens`
- `cost_estimate(prompt_tokens, completion_tokens)`

## Authentication & Secrets
- All credentials sourced from environment variables (`NERION_V2_OPENAI_KEY`, `NERION_V2_ANTHROPIC_KEY`, ...).
- Add `.env.example` documenting required keys; HOLO settings drawer surfaces "missing credential" warnings.
- Runtime refuses to start chat engine unless at least one primary provider is configured.

## Runtime Changes (Phase 2+)
1. Replace `app/chat/llm.py` with `LLMClientRegistry` that resolves provider based on intent (chat, code, planning, embeddings).
2. Introduce `app/chat/providers/` for adapter implementations (OpenAI, Anthropic, Gemini shims) backed by YAML config.
3. Add request budget manager: per provider concurrency limit, global timeout envelope (default 15s), exponential backoff.
4. Instrument latency + cost metrics, surface to HOLO signal highlights.
5. Update planner to rely on API-injected capabilities (`core/planner/planner_agent.py`).

## Configuration Updates
- `config/model_catalog.yaml`: point each logical role (chat, code, plan, embed) to API identifiers.
- `app/settings.yaml`: remove `ollama` toggles; add `default_provider`, `fallback_provider`, `max_cost_per_turn`.
- `scripts/setup_api_env.sh` (Phase 2): generate local `.env` with placeholders.

## Logging & Privacy
- Persist minimal request metadata: timestamp, provider, latency, token counts, cost estimate. No prompt logging by default.
- Offer opt-in transcript logging via `NERION_V2_LOG_PROMPTS=1`.
- Update docs to clarify data handling guarantees and provider policies.

## Migration Checklist
- [ ] Add compatibility layer so legacy CLI detects missing local models and points users to V2 repo.
- [ ] Ship `nerion v2 doctor` command to validate API connectivity and rate limit headroom.
- [ ] Document rollback path (how to return to Nerion classic for offline users).

## Timeline
1. **Phase 1 (this commit)** – Documentation + config scaffolding.
2. **Phase 2** – Implement provider adapters, remove local LLM dependencies, update planner/tooling.
3. **Phase 3** – HOLO UI telemetry, voice UX, signal highlights integration.
4. **Phase 4** – Testing hardening, release candidate, migration comms.
