# Repository Guidelines

## Repository Summary
Nerion V2 is a full-stack, self-improving developer agent composed of three tightly coupled cognitive layers: (1) the `selfcoder/` Behavioral Coach that plans, edits, tests, and applies code changes with policy guarantees; (2) the `nerion_digital_physicist/` research brain that models the codebase causally, designs curricula, and feeds new heuristics; and (3) the `app/` conversational layer plus the optional Holo Electron shell that mediate human interaction, approvals, and observability. Surrounding these cores are shared services (`core/`, `voice/`, `plugins/`), operational scripts (`scripts/`, `ops/`), configuration packs (`config/`, `.env`), and extensive test/fixture suites (`tests/`, `selfcoder/tests/`, `tests/digital_physicist/`). Artefacts emitted by one layer—plan manifests, coverage deltas, lesson outcomes—flow into the others via `out/`, `project_graph_cache.pt`, and learning stores, enabling continual improvement and necessitating synchronized documentation whenever capabilities evolve.

## Architecture & Project Layout
- `selfcoder/` is the automation core: planners, orchestrators, policy enforcement, learning loops, and CLI entry points (`cli/`, `planner/`, `policy/`, `tester/`).
- `app/` layers real-time UX (chat + voice) over the selfcoder runtime; `app/chat/` and `app/ui/` expose TUI bridges, while `app/config/` defines persona toggles.
- `core/` aggregates shared services (dialog manager, memory, HTTP tooling) consumed by both the chat surface and selfcoder pipeline.
- Execution scripts and ops runbooks live in `scripts/` (`run_local.sh`, `health.sh`) and `ops/`; artefacts and models land in `out/`, `export/`, and `digital_physicist_brain.pt`.
- Tests are split between `tests/` (public CLI, smoke, UX) and `selfcoder/tests/` (deep runtime coverage), with fixtures in `selfcoder/tests/fixtures/`. Docs reside in `docs/`, including planner grammar, policy DSL, and troubleshooting references.

## Component Guides
- `selfcoder/AGENTS.md` — deep dive on the automation core, planning loop, and reviewer gates.
- `nerion_digital_physicist/AGENTS.md` — curriculum learning, graph brain internals, and data stores.
- `app/AGENTS.md` — chat, voice, and UI bridge operations, including Holo integration.
- `app/ui/holo-app/AGENTS.md` — Electron shell, IPC contract, and UI build workflow.

## Agent Surfaces & Feature Map
- **Selfcoder Behavioral Coach:** Primary execution engine for refactors, diagnostics, and policy enforcement. Consumes intents from chat/CLI and returns vetted patches.
- **Digital Physicist:** Long-horizon researcher that maintains the causal graph, trains on self-authored lessons, and feeds heuristics back to the coach through shared learning stores.
- **Chat & Voice Layer:** Conversational gateway providing command shortcuts, offline voice control, and Opt-in Holo UI streams; brokers permissions and safety prompts.
- **Holo Shell:** Electron companion that visualizes agent cognition, session state, and interactive controls synced via JSONL IPC.

## Runtime Modes & Key Components
- The "Behavioral Coach" surfaces rapid heuristics for incremental work (see `selfcoder/planner/` and `selfcoder/policy/`).
- The "Digital Physicist" graph brain (`digital_physicist/`, `project_graph_cache.pt`) models causal structure; invoke via `scripts/run_learning_cycle.sh` for curriculum-driven self-improvement.
- Voice/UX agents route through `app/nerion_chat.py` and `voice/`; enable optional JS/TS bridges with `NERION_JS_TS_NODE=1` and the plugins in `plugins/`.

### Inter-Agent Workflow
- Chat captures intent → forwards structured commands to selfcoder planners; approvals loop back through chat safety prompts.
- Selfcoder revisions and telemetry are written to `out/` artefacts, which Digital Physicist jobs ingest during scheduled learning cycles.
- Curriculum updates from the Digital Physicist refresh caches (`project_graph_cache.pt`, `out/learning/`) that selfcoder loads on startup to bias future planning.
- Holo UI mirrors session state from chat while providing controls that ultimately route to selfcoder or policy modules.

## Environment Setup & Configuration
- Install with `pip install -e ".[dev,voice,web,docs]"` to enable tests, voice, web research, and documentation toolchains; minimal editing only requires `pip install -e .`.
- Copy credentials via `cp .env.example .env`; tune runtime toggles in `.env`, `config/profiles.yaml`, and `config/meta_policy.yaml`.
- Policy and intent routing are defined under `config/intents.yaml`, `config/tools.yaml`, and `config/self_mod_policy.yaml`; update alongside corresponding tests.
- Frequently used env switches: `NERION_POLICY` (`safe|balanced|fast`), `NERION_ALLOW_NETWORK`, `NERION_LEARN_ON_START`, and feature gates documented in `docs/CHEATSHEET.md`.

## Build, Test, and Validation Workflow
- `nerion-chat` launches the interactive terminal UI; `scripts/run_local.sh` starts the self-improvement harness with defaults.
- `pytest` runs the full suite; target by domain (`pytest tests/smoke -q`, `pytest selfcoder/tests/test_governor.py`).
- `selfcoder healthcheck --verbose` verifies policy wiring, tool registry, and dependency health before merging.
- `ruff check .` enforces lint; `ruff format path/to/file.py` handles code style. For JS/TS transforms, opt into Prettier or ESLint via the env flags mentioned above.
- Use `nerion patch preview plan.json` and `nerion patch safe-apply` when reviewing/landing agent-generated plans.

## Coding Standards & Quality Gates
- Python code follows PEP 8 with 4-space indentation, Ruff’s 100-character limit, and explicit type hints for new APIs (`pyproject.toml`).
- Keep planner grammars and policy DSLs (`docs/planner_grammar.md`, `docs/policy.md`) in sync with implementation changes; update docs when altering orchestrator behaviour.
- CLI verbs and module names stay snake_case; classes PascalCase; environment toggles SCREAMING_SNAKE. Prefix new scripts with verb-noun (e.g., `audit_*`).
- Do not modify excluded demo areas (`legacy_import/`, `src/demo.py`) without a design note; if changes are required, isolate them in dedicated commits.

## Testing Strategy
- Pytest auto-discovers `test_*.py` under `tests/` and `selfcoder/tests/`; integration flows sit in `tests/cli/`, `tests/smoke/`, and `selfcoder/tests/test_*integration.py`.
- Mark long-running voice checks with `@pytest.mark.voice`; CI skips them unless explicitly requested.
- Maintain `coverage.json` by adding unit tests when touching orchestrators, planners, or safety gates; regression tests are mandatory for policy or graph-learning changes.
- Use fixtures in `selfcoder/tests/fixtures/` to emulate planner outputs, tool manifests, and network responses without external calls.

## Commit & Pull Request Expectations
- Adopt the Conventional Commit prefixes seen in history (`fix(ci):`, `feat(planner):`, `docs:`) with imperative summaries ≤72 characters and detailed bodies when context is non-trivial.
- Branch names mirror `feature/<summary>`, `fix/<summary>`, or `docs/<summary>` per `CONTRIBUTING.md`.
- Pre-PR checklist: `ruff check .`, `pytest -q`, `selfcoder healthcheck --verbose`, plus targeted domain suites (voice, planner, JS bridges) when relevant.
- PR descriptions should link issues, outline architecture impact, list validation commands, and attach screenshots or terminal recordings for UX touches.
- Rebase against `main` to keep learning caches and policy snapshots aligned; avoid large mixed-topic PRs to preserve reviewer focus on safety-critical components.

## Security, Policy & Safety Controls
- Guardrails for file access and action types live in `docs/policy.md` and `config/meta_policy.yaml`; accompany any policy loosening with threat analysis and tests (`selfcoder/tests/test_policy_enforcement.py`).
- Secrets scanning is handled by the policy layer; never store credentials in repo—wire them through `.env` or platform-specific secret stores. Validate new tools against `config/tools.yaml` denylist entries.
- For networked features, keep `NERION_ALLOW_NETWORK` disabled by default and document opt-in flows; update `docs/troubleshooting.md` with mitigation steps for new failure modes.

## Troubleshooting & Support Resources
- Quick reference commands live in `docs/CHEATSHEET.md`; architecture deep dives are in `docs/agents.md`, `docs/policy_dsl.md`, and `docs/planner_grammar.md`.
- Use `scripts/health.sh` and `nerion voice devices` to debug voice pipelines; `docs/troubleshooting.md` outlines Node bridge pitfalls.
- Record experimental work and learning-phase adjustments in `docs/project_journal.md` to maintain institutional memory.
- When introducing new agent behaviours, summarize rationale and rollback plan in `docs/evolution_plan.md` or the relevant `docs/prs/` entry.

## Documentation Stewardship
- Any change that alters an agent’s responsibilities, capabilities, or workflows must update the corresponding `*/AGENTS.md` playbook in the same PR—rewrite the `Component Summary` (or equivalent) so it reflects the new state.
- Treat `AGENTS.md` files as canonical truth; reviewers should block merges when capability shifts lack doc updates.
- Capture transient experiments or feature flags in the relevant playbook until they are retired, then prune to avoid drift.
