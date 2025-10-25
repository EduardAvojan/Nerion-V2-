# Nerion V2 – API LLM Architecture

## Goals
- Deliver a responsive, reliable assistant by leaning on hosted LLM APIs instead of on-device checkpoints.
- Preserve Nerion differentiators (self-coding, planner tooling, HOLO UI) while redefining "privacy-first" as "user data never leaves the device without consent".
- Provide a clean migration path for existing Nerion users and a forward-looking contract for future integrations.

## Target Providers
| Priority | Provider | Use Case | Notes |
| --- | --- | --- | --- |
| P0 | OpenAI (gpt-5 / gpt-4o) | general chat + coder | strong tooling ecosystem, streaming support |
| P0 | Google (Gemini 2.5 Pro) | planner + multimodal | long context, vision input |
| P2 | Local gateway (optional) | privacy conscious deployments | shim for self-hosted vLLM or Ollama

Each provider registers through a thin adapter layer (`app/chat/providers` in Phase 2) that exposes:
- `async generate(messages, *, mode, metadata) -> Stream[Chunk]`
- `supports_tool_calls`
- `max_tokens`
- `cost_estimate(prompt_tokens, completion_tokens)`

## Authentication & Secrets
- All credentials sourced from environment variables (`NERION_V2_OPENAI_KEY`, `NERION_V2_GEMINI_KEY`, ...).
- Add `.env.example` documenting required keys; HOLO settings drawer surfaces "missing credential" warnings.
- Runtime refuses to start chat engine unless at least one primary provider is configured.

## Runtime Changes (Phase 2+)
1. Replace `app/chat/llm.py` with `LLMClientRegistry` that resolves provider based on intent (chat, code, planning, embeddings).
2. Introduce `app/chat/providers/` for adapter implementations (OpenAI, Google (Gemini) shims) backed by YAML config.
3. Add request budget manager: per provider concurrency limit, global timeout envelope (default 15s), exponential backoff.
4. Instrument latency + cost metrics, surface to HOLO signal highlights.
5. Update planner to rely on API-injected capabilities (`core/planner/planner_agent.py`).
6. Simulation harness bundles pytest, healthcheck, and optional lint/type/UI/regression checks; enable or skip each via `NERION_SIM_LINT_CMD`, `NERION_SIM_TYPE_CMD`, `NERION_SIM_UI_CMD`, `NERION_SIM_REG_CMD` (`skip` disables). Telemetry surfaces the per-check outcomes so operators see what blocked a rollout.
7. Phase 5 extends the security scanner (subprocess/dangerous FS/secret detectors) and enforces a repository allowlist for autonomous writes via `config/self_mod_policy.yaml`.
8. Telemetry snapshots now calculate apply success/rollback rates, governor/policy decisions, and per-window provider spend so HOLO/CLI dashboards can plot Nerion's self-evolution velocity.

## Configuration Updates
- `config/model_catalog.yaml`: point each logical role (chat, code, plan, embed) to API identifiers.
- `app/settings.yaml`: remove `ollama` toggles; add `default_provider`, `fallback_provider`, `max_cost_per_turn`.
- `scripts/setup_api_env.sh` (Phase 2): generate local `.env` with placeholders.
- `config/self_mod_policy.yaml` (Phase 5): default directory allowlist for autonomous self-mods; customise allow/deny patterns per deployment.

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
