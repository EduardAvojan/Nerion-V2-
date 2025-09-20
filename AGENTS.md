# Repository Guidelines

## Project Structure & Module Organization
- `selfcoder/`: Core engine, CLI (`nerion`, `selfcoder`), planners, analysis, tests under `selfcoder/tests/`.
- `app/`: Runtime app and chat entrypoints (`nerion-chat`), config, logging.
- `core/`, `plugins/`, `ops/`, `voice/`, `ui/cli/`: Planning/tooling, verified plugins, security/runtime, STT/TTS, terminal UI.
- `tests/`: Unit/CLI/smoke/UX/doc suites (`tests/unit`, `tests/cli`, `tests/smoke`, `tests/docs`, `tests/ux`).
- `scripts/`: Local helpers (e.g., `scripts/run_local.sh`). `config/`, `docs/`: YAML configs and docs.

## Build, Test, and Development Commands
- Setup (editable + dev tools): `pip install -e .[dev]`
- Run locally (agent): `bash scripts/run_local.sh` or `nerion-chat`
- CLI help: `nerion --help` • Health: `nerion healthcheck`
- Lint: `ruff check .` • Auto-fix: `ruff check . --fix`
- Tests (quiet, from pytest.ini): `pytest`
- Focus tests: `pytest tests/unit/test_file.py::TestClass::test_case -q`

## Coding Style & Naming Conventions
- Python 3.9+, 4-space indents, line length 100 (see `[tool.ruff]` in `pyproject.toml`).
- Use type hints and docstrings for public APIs.
- Naming: modules/vars/functions `snake_case`; classes `CamelCase`; constants `UPPER_SNAKE_CASE`.
- Keep patches minimal and scoped; avoid unrelated refactors.

## Testing Guidelines
- Framework: `pytest`; test discovery: files `test_*.py`, classes `Test*`, functions `test_*`.
- Suites live in `selfcoder/tests` and `tests/*` (unit/cli/smoke/ux/docs).
- Add tests for new behavior and regressions; include at least one smoke or CLI test when touching user flows.
- No strict coverage threshold enforced; prefer meaningful assertions over line coverage.

## Commit & Pull Request Guidelines
- Commits: imperative and scoped (e.g., `feat(core): add planner retries`, `fix(plugins): handle hash mismatch`).
- PRs: clear description, linked issues, rationale, before/after notes; include `nerion healthcheck` and relevant test output.
- Keep PRs small; document risk and rollback if touching security, plugins, or patching logic.

## Security & Configuration Tips
- Offline-first: do not add network calls without passing the security gate; respect `plugins/allowlist.json`.
- Opt-in networking only; when needed for dev, use explicit env toggles (e.g., `NERION_ALLOW_NETWORK=1`).
- Never hardcode secrets; prefer env vars and local configs in `app/settings.yaml`.

## Evolution Log
- 2025-09-16 — PR-P0: Crash-safe I/O, rotation, schema v3
  - Affected: `selfcoder/learning/continuous.py`, `app/logging/experience.py`
  - Prefs writes are atomic; schema bumped to v3 (`selfcoder/learning/jsonschema/prefs_v3.json`).
  - Log rotation via `NERION_LOG_ROTATE_BYTES` (default 52428800) for `out/experience/log.jsonl`.
  - Validate: `nerion learn review` updates `out/learning/prefs.json`; append several events and confirm rotation when size > threshold.
- 2025-09-16 — PR-P1: Robust learning (shrinkage + ESS cap)
  - Affected: `selfcoder/learning/continuous.py`
  - Per-intent success rates shrink toward global rates using effective sample size cap (`NERION_LEARN_EFF_N_MAX`, default 1000) to reduce flapping.
  - Top‑K hysteresis in selection (env: `NERION_LEARN_HYSTERESIS_M`, `NERION_LEARN_MIN_IMPROVEMENT_DELTA`) to avoid noisy swaps.
  - Validate: run `nerion learn review`; inspect `tool_success_rate_by_intent`, and watch stable preferred tool in prompts under small deltas.
- 2025-09-16 — PR-P2: A/B decisions + guardrails
  - New: `selfcoder/learning/abseq.py` (MSPRT), `selfcoder/learning/guardrails.py`.
  - `continuous.py` computes `experiments_meta` decisions and `guardrails` metrics/breach.
  - CLI: `nerion learn ab status --refresh` shows config, decisions, guardrails.
  - Env: `NERION_GUARDRAIL_ERR` (0.10), `NERION_GUARDRAIL_P95` (8000), `NERION_GUARDRAIL_ESC` (0.15).
- 2025-09-16 — PR-P3: Shadow runs + rollout scaffolding
  - New: `selfcoder/upgrade/shadow.py`; engine schedules non-blocking shadow replay after turns.
  - State written to `out/policies/upgrade_state.json` under `shadow` entries.
  - Env: `NERION_UPGRADE_SHADOW=1` to enable; `NERION_ROLLOUT_PCT` reserved for future staged rollout.
  - Validate: interact, then check `nerion learn ab status --refresh` (guardrails) and `out/policies/upgrade_state.json` for shadow metrics.
- 2025-09-16 — PR-P4: Observability + Replay/Export
  - Health: dashboard now shows ESS, realized epsilon, and intent drift KL.
  - Learn CLI: `replay --since 30d` and `export --window 30d --out out/learning/report.md`; `reset --intent X` added.
  - New: `selfcoder/learning/report.py` to summarize rotated logs and render markdown reports.
- 2025-09-16 — PR-P5: Contextual bandit hook (no behavior change)
  - `app/parent/driver.py` now passes a small `context` dict (intent, query_len, policy) to `build_master_prompt`.
  - `app/parent/prompt.build_master_prompt` accepts `context` (ignored for now) for future LinUCB/TS.
- 2025-09-16 — PR-P6: Privacy & Scope tags; merge policy
  - Prefs stats now include `scope: {user, workspace, project}`; env: `NERION_SCOPE_WS`, `NERION_SCOPE_PROJECT`.
  - Global-to-local merge honors scope: learning maps merge only when scope matches; `personalization` always merges.

## Electron HOLO UI Integration Notes (2025-09-22)
- `app/ui/holo-app/` houses the Electron shell; `src/main.js` spawns `python3 -m app.nerion_chat` with `NERION_UI_CHANNEL=holo-electron` so stdout/stdin act as the IPC bus.
- `app/chat/ipc_electron.py` maintains chat and command queues, lets Python register per-command handlers, and emits JSON events back to the renderer.
- `app/chat/ui_controller.py` coordinates incoming commands (memory, learning, upgrade, patch, artifacts, health, settings, PTT overrides) with core services:
  - Memory drawer actions operate on `SessionCache` and `LongTermMemory`, then emit `memory_session`, `memory_drawer`, `memory_update` events.
  - Learning panel pulls from `out/learning/` (`prefs.json`, `live.json`, `ab_status.json`) and serves diffs via `learning_diff` events.
  - Health commands trigger offline diagnostics and surface gate status tiles.
  - Upgrade offers are sourced from `app/learning/upgrade_agent.readiness_report`.
- `app/chat/engine.py` instantiates `ElectronCommandRouter` when the channel is enabled, registers Electron handlers, refreshes UI snapshots after each turn, and streams user/assistant messages to the renderer.
- Voice I/O now emits `chat_turn` events from TTS callbacks so the UI mirrors spoken replies in real time (`app/chat/voice_io.py`).
- Renderer bootstrap (`app/ui/holo-app/src/renderer.js`) requests initial snapshots (`memory`, `learning`, `upgrade`, `artifact`, `health`) once `window.nerion.ready()` fires, ensuring the dashboard reflects backend state immediately after launch.
