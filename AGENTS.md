# Agent Handbook – Nerion V2 (API-First)

> This document is the source of truth for any autonomous or assisted runs inside this repository. Read it before making changes. V2 pivots Nerion away from bundled local LLMs; all workflows below assume hosted providers.

## Mission & Guardrails
- **Goal:** deliver a responsive, self-improving assistant that fronts hosted LLM APIs while keeping user data under explicit control.
- **Privacy Stance:** nothing leaves the machine unless the user supplied the API key. Never log raw prompts/completions unless `NERION_V2_LOG_PROMPTS=1` is set.
- **Do Not Reintroduce** offline model pulls, Ollama launch agents, or legacy privacy claims unless the roadmap explicitly adds a “local gateway” adapter.

## Key References
- `docs/nerion-v2-api.md` – architecture, provider matrix, auth contract, migration checklist.
- `app/settings.yaml` – runtime defaults (`llm`, `credentials`).
- `.env.example` – template for required keys.
- `config/model_catalog.yaml` – API provider registry (`api_providers` node replaces local catalogs).

## Repository Layout (V2 context)
- `app/` – runtime entrypoints, chat engine, provider adapters (Phase 2 will add `app/chat/providers/`).
- `core/` – planner, tooling, memory glue; update when provider capabilities change.
- `selfcoder/` – self-improvement engine; keep compatible, but expect future simplifications once API tooling matures.
- `app/ui/holo-app/` – Electron HOLO shell; must surface API credential state, latency, cost.
- `tests/` & `selfcoder/tests/` – pytest suites; expand with mocks for remote APIs.
- `scripts/` – helper scripts; Phase 2 adds `setup_api_env.sh`, retire Ollama helpers.

## Runtime Expectations
1. Chat engine refuses to start unless at least one credential from `credentials.required` is available.
2. Provider selection happens via registry; respect per-role defaults (`chat`, `code`, `planner`, `embeddings`).
3. All network calls must flow through sanctioned adapters with retry/backoff + cost/latency logging.
4. UI components (HOLO, CLI) should display:
   - active provider
   - latency histogram / request cost
   - credential warnings when keys are missing
   - interactive provider selector that updates chat/code/planner overrides live

## Permissions & Safety Notes
- **Network:** V2 assumes outbound HTTPS to provider endpoints; sandbox/run configs must allow these domains. If sandbox denies, fail fast with a clear error.
- **Filesystem:** keep user secrets out of repo; `.env` stays ignored, `.env.example` is the only checked-in template.
- **Voice / Electron:** retain current PTT behavior; annotate when a feature now depends on network access (e.g., speech fallback prompts).

## Working Method (for Future Agents)
1. **Sync**: `git pull --rebase` (main) before starting work. If automation, fetch read-only.
2. **Branching**: feature branches named `feat/api-*`, `fix/ui-*`, etc. Do not work directly on `main`.
3. **Plan Tool**: required for multi-step edits. Describe steps and update as tasks finish.
4. **Implement**: touch only the scope declared in your plan. If you uncover unrelated bugs, open TODOs or issues instead.
5. **Format/Lint**: `ruff check .` (phase-in, once provider adapters are added). Respect `pyproject.toml` (max line length 100).
6. **Test**: minimally run `pytest tests/cli -q` + targeted suites impacted by your change. Use mocks for API calls (never hit live endpoints in CI without explicit user request).
7. **Review**: summarize diffs referencing file:line (e.g., `app/chat/engine.py:42`). Highlight risk areas, follow-up actions.
8. **Commit**: imperative subject (e.g., `feat(app): add openai provider adapter`). Include relevant test output in commit or PR description.

## Testing Matrix (Update as V2 stabilizes)
| Area | Command | Notes |
| --- | --- | --- |
| CLI basic | `pytest tests/cli -q` | ensures planner + prompts use new providers |
| Planner unit | `pytest tests/unit/test_parent_*.py` | update mocks for API outputs |
| Selfcoder smoke | `pytest selfcoder/tests/test_smoke.py` | confirm self-improve still works with remote models |
| HOLO UI | `npm test` (if added) / manual run | verify UI panels ingest API telemetry |
| Electron bridge | `npm run start` then `python -m app.nerion_chat --ui holo-electron` | ensure signal highlights, settings warnings fire |

## Release Checklist (draft)
1. Verify `.env` instructions and provider adapters work offline (mock servers) and online (staging keys).
2. Update README + docs to remove legacy offline claims.
3. Tag releases `v2.x.y`. Document migration path from classic (link to separate repo).
4. Publish changelog with cost/latency expectations and supported providers.

## Outstanding Work (Track & Update)
- [x] Implement provider registry and swap out `app/chat/llm.py` logic.
- [x] Rip remaining Ollama references (config, scripts, docs).
- [ ] Add API health diagnostics (`nerion v2 doctor`).
- [ ] Teach HOLO signal highlights to show latency + cost.
- [ ] Update CI to mock providers and ensure no secret leakage.

## Final Notes
- Treat this doc as living documentation. Update it whenever instructions change.
- If you encounter conflicting directives (user vs. handbook), escalate in chat, log assumptions, and capture follow-up items here for continuity.
