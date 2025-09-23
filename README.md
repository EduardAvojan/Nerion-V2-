# Nerion V2 (API-First, Voice-Forward, Self-Improving)

> **Nerion V2 notice (API-first pivot):** this repository tracks the evolution of Nerion toward hosted LLM providers. Follow `docs/nerion-v2-api.md` for the canonical migration plan and provider defaults. Sections below are being refreshed; when guidance conflicts, prefer the V2 docs.

Nerion V2 keeps the agent, planner, and safety loops on your machine while delegating language generation to the API providers you configure. You supply the API keys, Nerion streams prompts/responses through secure adapters, and the rest of the runtime (memory, learning, tools, UI) remains local and auditable.

Designed for modularity and explicit consent, Nerion remains a professional-grade platform that unites a voice-first assistant with a self-improving software engineer—now optimized for low-latency hosted models instead of bundled local weights.

## What is Nerion?
* **A Self-Hosted Orchestrator:** The agent, safety rails, and data stores live on your hardware. Outbound network traffic is limited to the LLM APIs you explicitly enable via environment variables.
* **An Autonomous Engineer:** Nerion can write, debug, and refactor its own code. You give it high-level goals in plain English, and it generates and safely applies the necessary code changes, complete with tests.
* **A Knowledge Engine:** Nerion can research topics on the web (when granted), read documentation, and synthesize information into artifacts—building a personalized knowledge base while respecting your configured trust boundaries.

---

## Quick Start (V2)

1.  **Install in Editable Mode**
    ```bash
    pip install -e .[dev]
    ```
    This links the `nerion` and `nerion-chat` commands to your checkout for fast iteration.

2.  **Provide API Credentials**
    ```bash
    cp .env.example .env
    # edit .env and add NERION_V2_OPENAI_KEY / NERION_V2_GEMINI_KEY / optional providers
    ```
    Nerion only contacts providers you configure. Missing keys are surfaced in the UI and `nerion doctor`.

3.  **Adjust Runtime Settings**
    * `app/settings.yaml` → tweak voice devices, default/fallback providers, and cost/timeouts.
    * `config/model_catalog.yaml` → review the provider registry (`api_providers`).

4.  **Launch the Agent**
    * Text/voice shell: `nerion-chat`
    * Electron HOLO UI: `npm run start` inside `app/ui/holo-app`
    * Legacy helper script (`scripts/run_local.sh`) still works but simply wraps `nerion-chat` in V2.

5.  **Interact with Nerion**
    * **Voice & Chat:** Hold **SPACE** to talk, or type `> your message` in the terminal.
      - Press and hold records the entire time until you release (true PTT).
      - Press **CapsLock** (or F9/F12) to toggle Speech ON/OFF with a clear on‑screen message.
      - Speaking indicators: terminal shows start/stop; HOLO UI receives events.
    * **CLI Tools:** Use the powerful `nerion` command for advanced operations:
        ```bash
        # Get a full list of commands
        nerion --help

        # Run a full system diagnostic
        nerion healthcheck

        # Ask Nerion to self-code a change with a natural language instruction
        nerion plan -i "add a try/except wrapper to the main processing function" --apply

        # Research a topic on the web (persist RAG-ready artifacts)
        nerion docs site-query --query "best laptop for developers 2025" --augment

        # Search your local knowledge chunks (offline)
        nerion docs search --topic "search:best laptop" --contains "battery" --limit 5

        # Inspect/clear network session & saved prefs (offline‑first gate)
        nerion net status
        nerion net prefs --clear

        # Inspect code graph and graph-aware rename preview
        nerion graph affected --symbol Foo --depth 2
        nerion rename --root . --old pkg.mod --new pkg2.mod --attr-old foo --attr-new bar --show-affected

        # Preview + patch (surgical apply)
        nerion patch preview plan.json
        nerion patch apply-selected plan.json --file a.py --file b.py
        # Hunk-level patching (apply only selected change hunks to a file)
        nerion patch preview-hunks plan.json --file a.py
        nerion patch apply-hunks plan.json --file a.py --hunk 0 --hunk 2
        # Safe apply (gate + subset tests + apply if clean)
        nerion patch safe-apply plan.json --file a.py --file b.py
        # Minimal colored diff preview
        nerion patch preview plan.json --simple-diff
        # Interactive TUI to pick hunks and apply
        nerion patch tui plan.json
        # In the TUI:
        #  - SPACE: toggle current hunk
        #  - g: run security gate on the current selection (risk score + top findings)
        #  - t: run a quick pytest subset in a shadow repo (uses tests/smoke if present; else full)
        #  - J: show only JS/TS findings; E: only ESLint; T: only tsc; ]/[ navigate files with findings
        #  - I: show JS/TS affected importers for changed files (depth 1)
        #  - R: show Python rename affected files (depth 2) when the plan contains rename_symbol
        #  - a: apply only if gate passes and subset is green (auto‑runs subset if needed)
        #  - A: force apply (override gate/subset failures)
        # Env:
        #  - NERION_PATCH_TUI_NODEIDS="tests/test_x.py::test_y,tests/test_z.py::test_w" (optional subset override)
        #  - NERION_PATCH_TUI_PYTEST_TIMEOUT=300 (seconds)
        # Persisted prefs: overlay filters, last profile, last subset nodeids

        # Verify plugins against allowlist and hashes
        nerion plugins verify

        # Force offline mode for a session window
        nerion net offline --for 30m

        # Lint and doctor
        nerion lint
        nerion lint --fix
        nerion doctor
        # Health dashboards
        nerion health dashboard
        nerion health html --out out/health/index.html --refresh 10

        # Health and latency summaries
        nerion health dashboard
        nerion voice metrics --last 100

        # Policy (allow/deny/limits)
        nerion policy show
        nerion policy audit --json

        # Artifacts export (Markdown report)
        nerion artifacts export --topic "search:laptops" --out out/report.md

        # Trace recent runs (action timings)
        nerion trace last --last 20
        # Local HTTP server (IDE integration endpoints)
        nerion serve --host 127.0.0.1 --port 8765
        # Endpoints: GET /version; POST /patch/preview, /patch/apply, /review (all return {ok,data,errors})

        # Print build version tag
        nerion --version

        # Benchmark/repair a standalone task directory (tests inside)
        # Auto profile selection (bench-recommended) is used by default
        nerion bench repair --task /path/to/task --max-iters 6
        ```

## Telemetry & Reflection Fabric (Phase 1)

Nerion now ships a live telemetry backbone so the assistant, operators, and future self-improvement jobs share a single source of truth.

* **Unified event bus:** Every chat, planner, model, and tooling event publishes through the new telemetry bus. Events are durably persisted to `out/telemetry/events.sqlite`, eliminating the need to scrape JSONL logs.
* **Snapshotting:** `ops.telemetry.snapshots.write_snapshot()` captures git state, active provider defaults, and recent event totals under `out/telemetry/snapshots/` for time-travel audits.
* **Reflection pipeline:** Run `nerion telemetry reflect` to summarise the latest window (counts, provider latency/cost, anomaly hints, tag clusters). Reports live in `out/telemetry/reflections/`; add `--embed` to persist highlights into the vector store (`out/memory/vector_store.sqlite`).
* **Scheduling helper:** `nerion telemetry schedule --at tonight --embed` prints cron/launchd examples so reflections can refresh automatically and feed the vector store without manual intervention; the generated command now chains a `nerion telemetry knowledge` export (disable with `--no-knowledge`).
* **Operator surfaces:** The CLI metrics capsule and HOLO dashboard ingest `TelemetryStore` snapshots, showing active providers, request volume, spend, and current anomalies with inline call-outs. Reflection digests link from the same panels for quick triage.
* **Memory lessons:** Reflections now pull recent memory journal events so the snapshot highlights what Nerion learned or promoted (`summary.memory`, `lessons` block). The metrics bar shows total memory touches alongside provider stats.
* **Experiment journal:** Track hypotheses and outcomes with `nerion telemetry experiments log|update|list`; entries live under `out/telemetry/experiments.json` and can link back to specific reflections.
* **Knowledge graph:** `nerion telemetry knowledge` fuses git churn, dependency analysis, and telemetry failures into `out/telemetry/knowledge_graph.json`. Use `--json` for the full graph (components, edges, hotspot scores) or feed the artifact into downstream planners.

## Architect Brief Generator (Phase 3)

`nerion architect` distils telemetry hotspots, coverage gaps, static-analysis smells, and outstanding roadmap tasks into structured upgrade briefs. Each brief supplies rationale, acceptance criteria, and priority weighting so the planning loop can focus on the riskiest components first. Pass `--json` to integrate the briefs into automations, or tweak inputs with flags like `--window` and `--no-smells`.

- **Policy-aware prioritiser:** Briefs are now scored against your active policy (`safe`/`balanced`/`fast`) with auto/review/block decisions based on risk, effort, and an optional cost budget (`NERION_ARCHITECT_COST_BUDGET`). The JSON output exposes the gating reasons so automation can respect review holds.
- **Planner context bridge:** `nerion plan`, the self-coding voice loop, and `plan_with_llm` automatically load the top matching brief and feed its signals (component, rationale, anomalies, suggested targets) into the planner prompt. Plans embed the brief metadata under `metadata.architect_brief` so executors and dashboards share the same context.

## Apply Policies & Verification (Phase 4)

- **Autonomous apply gating:** `nerion plan --apply` now checks the architect brief decision (auto/review/block) before writing to disk. Review- and block-gated plans exit early unless you pass `--force-apply`; the CLI prints the policy reasons so operators can respond accordingly. Self-improve runs respect the same policy—set `NERION_SELF_IMPROVE_FORCE=1` or `NERION_SELF_IMPROVE_ALLOW_REVIEW=1` to override when running unattended.
- **Shared policy helper:** `selfcoder.planner.apply_policy.evaluate_apply_policy()` exposes the gating metadata so other automations (voice loop, HOLO triggers, batch apply) can make consistent decisions based on the active policy profile. Voice self-coding and the learning upgrade agent now consult the same gate (`NERION_UPGRADE_FORCE=1` / `NERION_UPGRADE_ALLOW_REVIEW=1` opt into overrides) before touching the repo.
- **Expanded post-apply verifier:** After a successful apply, Nerion runs healthchecks plus any optional commands you configure via environment variables:
  - `NERION_VERIFY_SMOKE_CMD`
  - `NERION_VERIFY_INTEGRATION_CMD`
  - `NERION_VERIFY_UI_CMD`
  - `NERION_VERIFY_REG_CMD`

  Each command honours matching `*_TIMEOUT` overrides (seconds). Failures automatically trigger rollback in CLI/Selfcoder apply flows, and telemetry logs capture the exit status for dashboards.
- **Smart defaults:** When those environment variables are unset, Nerion auto-detects standard suites—running `pytest tests/smoke -q`, `pytest tests/integration -q`, or `npm run build --prefix app/ui/holo-app` when the corresponding folders exist. Override or disable with the env vars above so CI can pin explicit commands.
- **Pinned CI defaults:** `.env.example` seeds the verifier commands with `python -m pytest tests/smoke -q` and `python -m pytest tests/cli -q` while leaving UI/regression at `skip`. Update those entries (and matching CI env exports) to reflect your organisation’s pipelines so local runs and automation stay aligned.

## Governor Controls (Phase 5)

- **Centralised execution governor:** All autonomous apply flows (self-improve, `nerion plan --apply`, the learning upgrade agent, and the voice self-coding loop) now consult `selfcoder.governor.evaluate()` before touching the repo. The governor enforces a minimum spacing between runs, hourly/daily rate caps, and configurable execution windows.
- **Configuration knobs:** Tune the behaviour via `NERION_GOVERNOR_MIN_INTERVAL_MINUTES`, `NERION_GOVERNOR_MAX_RUNS_PER_HOUR`, `NERION_GOVERNOR_MAX_RUNS_PER_DAY`, and `NERION_GOVERNOR_WINDOWS` (`HH:MM-HH:MM` CSV). Drop a `config/governor.yaml` alongside other configs to persist team defaults; per-run overrides come from the same environment variables.
- **Human override hooks:** Operators can bypass the governor with `--force-governor` (CLI plan/apply) or the existing force flags (`NERION_SELF_IMPROVE_FORCE`, `NERION_UPGRADE_FORCE`, `NERION_PLAN_FORCE_GOVERNOR`, etc.). Setting `NERION_GOVERNOR_OVERRIDE=1` unlocks unattended maintenance windows when the caps would otherwise block.
- **Stateful telemetry:** Governor decisions are recorded under `out/governor/state.json` and tagged in telemetry events (`kind=governor`) so dashboards surface when throttles trip. Successful applies record executions into the same ledger, enabling precise sliding-window enforcement without querying the full telemetry store.

## Security & Policy Enhancements (Phase 5)

- **Self-mod allowlist:** `config/self_mod_policy.yaml` ships with a directory allowlist so autonomous plans stay within vetted areas of the repo. Extend or tighten the list per project needs (format matches the existing policy DSL). The policy loader now falls back to this file automatically when `.nerion/policy.yaml` is absent, so every apply path inherits the directory guardrails out of the box.
- **Stronger detection rules:** `selfcoder/security/rules.py` now flags `subprocess.Popen`/`asyncio.create_subprocess_*`, destructive `os.remove`/`pathlib.Path.unlink`, and adds regex sweeps for Slack/GitHub tokens. Findings surface through `selfcoder.security.gate` so preflight reviews and CI fail before risky code lands.

## Evolution Metrics Dashboard (Phase 5)

- **Telemetry snapshot upgrades:** `ops/telemetry/operator.load_operator_snapshot()` aggregates apply success rate, rollback counts, and governor/policy decisions alongside existing prompt/latency stats. Total provider spend for the current window is calculated from provider metrics and exposed to downstream surfaces.
- **Operator surfaces:** The CLI dashboard (`nerion health dashboard`) prints the new apply/policy/governor summaries, and the HOLO signal highlight capsule inherits the richer metrics payload so velocity and rollback trends are visible at a glance.

### Quick Sanity Checks (Provider Routing)

- Verify the active chat provider:
  ```bash
  nerion profile explain --task chat | jq .env.NERION_V2_CHAT_PROVIDER
  ```
- Run a dry code completion via the provider registry:
  ```bash
  python - <<'PY'
  from app.chat.providers import get_registry
  print(get_registry().generate(role='code', prompt='Reply OK only.').provider)
  PY
  ```
- Inspect router decisions in verbose mode:
  ```bash
  NERION_ROUTER_VERBOSE=1 nerion plan --llm -f demo.py -i "print('hi')"
  ```

### Offline Agent Improvements (New)
- Built-in repair proposer (no cloud): proposes surgical patches from failing logs (imports, NameError typos → safe rename, None/Index/Type/Value errors, numeric tolerance). Integrated shadow‑eval promotes green candidates. Used automatically when no plugin is present.
- Planner strictness + cache: `nerion plan --llm --json-grammar` enforces JSON plan output (schema-validated) and caches normalized plans in `.nerion/plan_cache.json`.
> For a compact command reference, see the Cheat Sheet: `docs/CHEATSHEET.md`.

---

## Features

### Core Architecture
-   **Local Control & Explicit Network:** All core logic, memory, and sensitive data processing stay on your device; outbound LLM calls only target providers you configure.
-   **Sovereign & Controllable:** You own the agent, the code, and the data. All major functions are controllable via CLI and configuration files.
-   **Modular & Extensible:** A secure plugin system allows for adding new tools, commands, and self-coding capabilities.
-   **Resilient by Design:** Features defensive programming and graceful degradation to ensure the agent remains operational even if optional components fail.

### Autonomous Capabilities
-   **Safe Self-Coding Engine:** Uses Abstract Syntax Tree (AST) editing for precise and syntactically correct code modifications, not fragile text replacement.
-   **Safety & Rollback Guard:** Automatically snapshots the codebase before any change. If a post-change healthcheck or test fails, it automatically reverts, ensuring the agent never permanently breaks itself.
-   **Proactive Self-Improvement:** Can be configured to periodically scan its own codebase for "code smells," vulnerabilities, or areas for improvement, and will autonomously generate plans to upgrade itself.
-   **Enhanced Simulation Harness:** `--simulate` now captures pytest, healthcheck, and optional lint/type/UI/regression checks in one report. Configure extra steps with `NERION_SIM_LINT_CMD`, `NERION_SIM_TYPE_CMD`, `NERION_SIM_UI_CMD`, and `NERION_SIM_REG_CMD` (set to `skip` to disable). Results flow into artifacts/telemetry so dashboards and planners inherit the same readiness signals.
-   **Automated Dependency Management:** Can scan for outdated or insecure package dependencies, plan an upgrade path, and safely apply the updates within a sandboxed simulation before finalizing.
-   **General Knowledge Assimilation:** A powerful, domain-agnostic engine that can research topics, read documentation, and synthesize information from the web. It uses a "Scout" (Search API) and "Analyst" (Web Reader) pipeline to build a rich, personalized knowledge base.
-   **Advanced Memory System:** Multi-layered memory now ships with namespaced schema v2 storage (optional encryption-at-rest), append-only journaling, hybrid semantic retrieval (BM25 + embeddings), PII/prompt-poison quarantine, automated consolidation with utility-based pruning, and a persistent session cache that promotes repeated lessons into long-term memory.
 -   **Planner‑Driven Multi‑Step Execution:** The Parent planner can return ordered multi‑step plans; the executor now supports per‑step timeouts and `continue_on_error` for robust long workflows, with visible “Step N/M…” progress.

### Self‑Learning & Upgrades
-   **Robust Continuous Learning:** Shrinks per‑intent tool success rates toward global rates with an effective sample cap, and applies Top‑K hysteresis to avoid noisy flapping.
-   **Crash‑Safe Prefs & Logs:** Atomic JSON writes for learning prefs (v3 schema) and size‑based rotation for experience logs.
-   **Experimentation & Guardrails:** Sequential A/B decisions (mSPRT) with guardrails on error rate, p95 latency, and escalation; surfaced via CLI for operators.
-   **Safe Self‑Upgrade:** Shadow replays run in the background and record metrics without affecting live traffic; staged rollout knob enables canary → gradual rollout when healthy.
-   **Observability:** Health dashboard shows effective sample size (ESS), realized exploration rate (ε), and intent distribution drift (KL). Replay/export tools make runs reproducible.

Common self‑learning commands:

```bash
# Update learned prefs from recent logs and inspect
nerion learn review
nerion learn show --intent code_fix --explain

# A/B controls and status (decisions + guardrails)
nerion learn ab start --name demo --split 0.5 --arms baseline,treatment
nerion learn ab status --refresh
nerion learn ab stop

# Replay/export for ops
nerion learn replay --since 30d
nerion learn export --window 30d --out out/learning/report.md
nerion learn reset --intent code_fix

# Health dashboard (HTML)
nerion health html --out out/health/index.html --refresh 10
```

### Interfaces & Tooling
-   **Conversational Interface:**
    -   **Voice-First:** Supports Push-to-Talk (PTT) and VAD modes with local, streaming STT for a responsive experience.
    -   **Barge-In:** Allows the user to interrupt the agent mid-sentence for a natural conversational flow.
    -   **Text Chat:** A fully integrated terminal chat interface for when a microphone is unavailable.
    -   **Busy Indicator:** Friendly, auto-clearing spinner appears only for long operations (LLM thinking, planning, web steps); never for instant answers.
-   **Comprehensive CLI:** A professional-grade command-line interface provides access to all of Nerion's core functions:
    -   **`plan`, `apply`, `simulate`:** The main interface for the self-coding engine.
    -   **`self-improve`, `deps`, `audit`:** Tools for proactive maintenance and dependency management.
    -   **`docs site-query`, `summarize`:** The interface for the knowledge assimilation engine.
    -   **`memory`:** Inspect or search long-term memory directly (`nerion memory show --k 10`, `nerion memory search "likes coffee"`).
    -   **`voice`:** Tools for configuring and diagnosing the voice stack.
    -   **`healthcheck`, `diagnose`:** Robust diagnostic tools for ensuring system integrity.
-   **Developer Mode:** Ask “switch to dev mode” to see capability options (self‑code, self‑improve, memory, web‑research, voice‑text, security, planner, tools‑plugins) and request detailed, dynamic summaries (e.g., “details for self‑code”).
-   **Dynamic Capability Summary:** “What can you do?” produces a first‑person, auto‑updating summary derived from the live tool/intent/plugin configuration.
-   **Command Routing Guardrails:** Action verbs (run/scan/check/diagnose/rename/…​) route to tools with a short planning window; generic prose fallback is disabled for these commands.
-   **Direct‑Run System Tasks:** Common offline tasks (health check, diagnostics, smoke tests) run immediately with an acknowledgement and spinner—no planner dependency.
-   **Mapped Planner Tools (zero‑shot):** `read_url`, `web_search`, `run_healthcheck`, `run_diagnostics`, `list_plugins`, `run_pytest_smoke`, `rename_symbol` (Pydantic‑validated arguments; allow‑listed).
 -   **Local File Tools (safe):** `read_file` and `summarize_file` read and summarize text files within the repo jail; large files are truncated safely.
 -   **Concise Mode:** Toggle “concise mode on/off” to keep general responses to a single sentence by default.
 -   **Cancel Long Operations:** Say “cancel” to abort a multi‑step plan between steps.
-   **Automated Testing Tools:** Includes an `autotest` mode to automatically generate test scaffolds for new code and a `coverage` checker to ensure code quality is maintained.

Key environment flags (optional):

```bash
# Learning stability
export NERION_LEARN_EFF_N_MAX=1000
export NERION_LEARN_HYSTERESIS_M=3
export NERION_LEARN_MIN_IMPROVEMENT_DELTA=0.02

# Guardrails
export NERION_GUARDRAIL_ERR=0.10
export NERION_GUARDRAIL_P95=8000
export NERION_GUARDRAIL_ESC=0.15

# Upgrades & rollout
export NERION_UPGRADE_SHADOW=1
export NERION_ROLLOUT_PCT=0.00

# Prefs/logs hygiene
export NERION_LOG_ROTATE_BYTES=52428800

# Scope (privacy)
export NERION_SCOPE_WS=default
export NERION_SCOPE_PROJECT=default
```

## Security
-   **Filesystem Jail:** All operations are strictly confined to the project directory to prevent unauthorized file access.
-   **Safe Subprocess Wrapper:** All external commands are executed through a hardened wrapper that prevents shell injection and enforces timeouts.
-   **Plan Schema & Security Validation:** All self-coding plans are validated against a strict schema before execution to prevent malicious or malformed instructions.
-   **Static Preflight Fusion (Unified Gate):** Predicted edits are checked by an offline gate that combines a fast AST/regex scanner with optional local tools (ruff, mypy, bandit, semgrep). Findings are normalized into a risk score; policy profiles (`safe|balanced|fast`) set thresholds (safe blocks aggressively; fast degrades to core checks only). `nerion doctor` includes a summary row (risk score and counts) for quick overview.
-   **Policy DSL (Repo‑Configurable):** Define allow/deny for actions, allowed/denied paths, and size limits in `.nerion/policy.yaml` or `config/policy.yaml`. The orchestrator and security gate enforce policy before applying changes. Use `nerion policy show` to view the merged policy and `nerion policy audit` to dry‑run path/limit checks.
-   **Plugin Allowlist & Hash Pinning:** Only explicitly allowlisted plugins can be loaded; plugin files must reside under `plugins/` (no symlinks). Optionally pin plugin file hashes in `plugins/allowlist.json` → verify with `nerion plugins verify`.
-   **Secret-Redacting Logger:** Automatically scrubs sensitive information (API keys, passwords) from logs and artifacts.

---

## Recent Enhancements

These improvements were added to strengthen reliability, clarity, and developer ergonomics.

### Routing & Tools
- Balanced, offline‑first routing: local intents first; planning is engaged only when needed.
- Command verbs (run/scan/check/diagnose/rename/…​) always plan tools (5s cap) — no generic chat for commands.
- Direct‑run system tasks: healthcheck and diagnostics return fast with clear acknowledgements and spinners.

### Knowledge & RAG
- Web search and site‑query now persist a compact knowledge index and per‑URL chunks in `out/knowledge/`.
- Query your local knowledge offline via `nerion docs search --topic "search:..." --contains "..." --limit 5`.

### Graph & Refactors
- Inspect defs/uses for any symbol via `nerion graph affected --symbol Foo`.
  - Use `--depth N` to include N-hop reverse importers; add `--methods` to include class methods (as `Class.method`); add `--json` for JSON output.
- Graph‑aware rename preview: `nerion rename ... --show-affected` prints defs/uses plus transitive affected file counts via a lightweight import graph.
- Cached code index: Nerion builds and uses a cached index at `out/index/index.json` (mtime‑based) for fast symbol defs/uses/imports and transitive `affected()` queries.
  - Parallel parse: large repos automatically use a parallel index build; set `NERION_INDEX_WORKERS` to override worker count. Stats (`files`, `built_ms`) are stored alongside the index.
- Better targeting: the built‑in repair proposer consults the index/graph as a fallback to pick focus files when stack traces/greps are inconclusive.

### Parent Planner Insight
- Parent steps may include `why` and `cost_hint_ms`; executor logs `duration_ms` and surfaces concise acks like “Step N/M: tool — why (~Xms)”.

---

## Docs & How‑Tos

- Policy DSL quick guide: see `docs/policy_dsl.md` for action/path/limit examples and the `nerion policy` commands.
- Patch TUI & IDE bridge: see `docs/ide_bridge_tui.md` for `nerion patch tui` usage and the local HTTP endpoints.
- JS/TS Node Bridge: see `docs/js_ts_node_bridge.md` to enable the ts‑morph runner (`npm install ts-morph`, `NERION_JS_TS_NODE=1`).

### Voice UX
- Adaptive speaking rate based on utterance length; live TTS tuning via `/say rate 180` and `/say voice Daniel`.
- Mic calibration duration configurable via `NERION_CALIBRATE_SECS`.

### Voice Diagnose + VAD Tips
- Quick diagnostics: checks mic access and runs a short VAD monitor with a simple amplitude bar.
  - `nerion voice diagnose --duration 10 --device "Studio Display Microphone" --vad-sensitivity 8 --min-speech-ms 120 --silence-tail-ms 250`
- List input devices (IDs/names/channels):
  - `nerion voice devices`
- Enable VAD + barge‑in mode (interrupt TTS on speech):
  - `nerion voice set --mode vad --barge-in --device "Studio Display Microphone"`
  - Show current settings: `nerion voice show`
- STT backend (offline‑first): set via env
  - `NERION_STT_BACKEND=whisper|whisper.cpp|vosk|sphinx|auto`, `NERION_STT_MODEL=tiny|small|base|…`
  - `WHISPER_CPP_MODEL_PATH=/path/to/ggml-or-gguf.bin` (for whisper.cpp binding if selected)
  - Force offline and fail closed: `NERION_STT_OFFLINE=1`, `NERION_STT_STRICT_OFFLINE=1`
- Latency metrics: `nerion voice metrics --last 100` (reads `out/voice/latency.jsonl`)
- Optional deps: `webrtcvad` + `pyaudio` for VAD/barge‑in; `pocketsphinx` (sphinx); `vosk` with `VOSK_MODEL` dir; `whisper` (local model).

### Voice Quickstart
- Pick a TTS backend (offline):
  - Quick: `nerion voice set --backend pyttsx3` (cross‑platform) or `nerion voice set --backend say` (macOS)
  - Optional high‑quality: Piper or Coqui (requires local CLIs/models)
    - `export NERION_TTS_BACKEND=piper && export PIPER_MODEL_PATH=/path/to/model.onnx`
    - `export NERION_TTS_BACKEND=coqui && export COQUI_MODEL_PATH=/path/to/model.pth`
- Pick an STT backend (offline):
  - `export NERION_STT_BACKEND=whisper|whisper.cpp|vosk|sphinx|auto`
  - whisper.cpp: `export WHISPER_CPP_MODEL_PATH=/path/to/ggml-or-gguf.bin`
- Choose a microphone and tune VAD:
  - List devices: `nerion voice devices`
  - Persist settings: `nerion voice set --mode vad --barge-in --device "Studio Display Microphone" --vad-sensitivity 8`
  - Diagnose: `nerion voice diagnose --duration 10 --device "Studio Display Microphone" --vad-sensitivity 8`
- Start chatting with voice: `nerion-chat` (hold SPACE to talk; CapsLock toggles speech)

### Memory Quality
- “pin that for N days” attaches a TTL; prune honors expiry (age ≥ N days).
- Naive topic/sentiment tags (e.g., “food”, “tools”, “work”, “positive/negative”) improve recall ranking.

### Net & Version
- `nerion net status` / `nerion net prefs --clear` to inspect/clear network gate state.
- `nerion net offline --for 30m` to force offline mode for a session window (suppresses prompts).
- `nerion --version` prints the build tag.
- Planner tool mapping extended: `run_diagnostics`, `list_plugins`, `run_pytest_smoke` added alongside existing tools.
- Multi‑step executor now supports per‑step timeouts and `continue_on_error`, with visible “Step N/M” acks.

### Voice & UX
- True hold‑to‑talk: audio is captured until you release SPACE (no early cut‑off).
- Reliable toggles: CapsLock (or F9/F12/Ctrl+F9) toggles Speech ON/OFF with explicit messages.
- STT normalization: corrects common name/phrase misrecognitions (e.g., “marion”→“Nerion”, “self approved”→“self improve”).
- Identity guardrails: answers identify only as “Nerion” and avoid model/provider claims.
- Busy indicator: subtle spinner appears only for tasks that actually take time and clears automatically.
- Action feed: “show last actions” lists recent tool calls with status and duration for fast troubleshooting.
 - Concise mode: one‑sentence answers on demand (see below); easy to toggle by voice or text.
 - Cancel: say “cancel” to stop long multi‑step runs between steps.

### Self‑Coding & Safety
- Safer multi‑file workflows: step acks, timeouts, and optional continue‑on‑error in plan execution.
- Healthcheck runner hardened: concise OK/FAIL output; no unpacking errors.

### Memory & Answers
- Humanized memory recall: returns “You like …” style facts instead of parroting raw inputs.
- Dynamic capability summary: first‑person, live view of abilities that updates as tools/plugins change.

### Testing & CI

### Learning & Biasing (Contextual)
- Per‑intent learning: `nerion learn review` now computes per‑intent tool success in addition to global rates.
- New prefs fields in `out/learning/prefs.json`:
  - `tool_success_rate_by_intent[{intent}][tool]` — smoothed success rates by intent.
  - `tool_sample_weight_by_intent[{intent}][tool]` — weighted sample counts by intent.
  - Global aliases: `tool_success_rate_global`, `tool_success_weight_global`.
- Parent biasing honors per‑intent rates when detectable (via triage or `NERION_INTENT_HINT`); falls back to global.
- Bandit strategy knob: `NERION_BANDIT_STRATEGY=greedy|ucb|thompson` with `NERION_BANDIT_UCB_C` for UCB bonus; still supports `NERION_BANDIT_EPSILON`.

### LLM Providers & Provisioning (V2)
- Provider registry lives in `config/model_catalog.yaml` under `api_providers`.
- Runtime defaults (`app/settings.yaml`) define `llm.default_provider`, `llm.fallback_provider`, request timeouts, and per-turn cost ceilings.
- Environment overrides:
  - `NERION_V2_DEFAULT_PROVIDER` – primary provider when a role-specific env var is unset.
  - `NERION_V2_CHAT_PROVIDER`, `NERION_V2_CODE_PROVIDER`, `NERION_V2_PLANNER_PROVIDER`, `NERION_V2_EMBEDDINGS_PROVIDER` – per-role pins.
  - `NERION_V2_REQUEST_TIMEOUT` – global override for request timeout (seconds).
- CLI helpers:
  - `nerion models ensure --provider openai:gpt-5` – set code provider for the session.
  - `nerion models bench --providers google:gemini-2.5-pro openai:gpt-5` – measure latency (mocked in CI).
  - `nerion models router --task code -f demo.py -i "add logging"` – print the router’s provider decision.
- Diagnostics: `nerion doctor` and HOLO’s health dashboard list missing credentials, quota errors, and recent latency samples.

#### Task-Aware Router & Profiles (V2)
- Router emits `(provider, model)` pairs and never autoprovisions local weights.
- Integration points (`plan --llm`, chat engine, patch safe-apply, rename, batch) consult the provider envs before execution.
- Profiles (`config/profiles.yaml`) influence env vars:
  - `llm.default_provider` → `NERION_V2_DEFAULT_PROVIDER`
  - `llm.fallback_provider` → `NERION_V2_FALLBACK_PROVIDER`
  - `llm.roles.chat|code|planner|embeddings` → `NERION_V2_{ROLE}_PROVIDER`
  - `net.allow: true` → `NERION_ALLOW_NETWORK=1`
- Verbose router logging: set `NERION_ROUTER_VERBOSE=1` for one-line provider summaries; enable `NERION_ROUTER_LOG=1` to persist JSON lines under `.nerion/router_log.jsonl`.

### Built‑in Repair Proposer (New)
- Offline fixers: unresolved import, NameError aliases (np/pd), None/Index/Type/Value guards, numeric tolerance; surgical diffs only.
- Shadow eval: validate on a failing subset then full run; promote green candidates.
- Graph integration: when logs are sparse, proposer leverages the cached index to locate likely impacted files.

User vs Developer mode:
- User mode (default): orchestrator sets smart defaults (auto‑select, strict JSON) and handles provisioning prompts; no env setup required.
- Developer mode: set `NERION_MODE=dev` to opt out and manage env/config manually.
- Per‑step credit assignment: in multi‑step plans, the terminal tool receives full credit (1.0) while prior steps get partial credit (default 0.25). Optional cost weighting penalizes long‑duration steps (`NERION_LEARN_COST_WEIGHT=1`).
- A/B evaluation (local, offline): `nerion learn ab start/stop/status` controls a local `out/learning/ab.json` assignment spec. The experience log records `experiment` metadata, and `nerion learn review` aggregates per‑arm success/latency into `prefs['experiments']`.
- Confidence gating: the Parent biasing uses Wilson intervals and a delta threshold. If per‑intent samples are low or top candidates' intervals overlap, biases are softened or fall back to global; header shows “softened (low confidence)”.

### Safety, Policy, Migration (Learning)
- Preferences: per‑repo + global. Repo prefs live under `out/learning/prefs.json`. A global fallback lives under `~/.nerion/prefs_global.json` and is merged as defaults (local values win). Schema versioning adds `schema_version` and `last_migrated`.
  - Auto‑migration: older prefs are loaded and stamped on next `nerion learn review`; missing fields are handled gracefully.
  - Reset: `nerion learn reset` removes prefs; the next review recreates them with the current schema.
- Policy interplay (`NERION_POLICY`):
  - `safe`: requires more per‑intent samples and a larger delta before applying strong per‑intent bias; bias often softens until confidence rises.
  - `balanced`: default confidence thresholds.
  - `fast`: allows smaller deltas for biasing and keeps prompts compact (smaller top‑K; exploration favored).
- Network safety: Learning, biasing, and the self‑learn dataset/training stubs run entirely offline and respect the global `NERION_ALLOW_NETWORK` gate.
- Additional unit tests for executor timeouts/continue‑on‑error, healthcheck formatting, STT normalization, step acks, and planner tool mapping.
- Pytest configuration now collects both `selfcoder/tests` and `tests/unit` by default.

### Additional Enhancements
- Preview + patch mode: unified diff preview per plan and a CLI to apply selected files, plus hunk-level apply for surgical edits to a single file.
- Unified‑diff fallback (safe): new plan action `apply_unified_diff` lets you apply minimal text diffs when AST transforms are insufficient. All changes are repo‑jailed and go through Reviewer + Security gates.
 - Repair runner: `nerion bench repair --task <dir>` triages failing tests, writes artifacts under `out/bench/<task>/`, and runs an iterative repair loop. If no local proposer plugin is found at `plugins/repair_diff.py`, Nerion automatically uses the built‑in offline proposer.
  - Proposer architecture: multiple candidate diffs per iteration (heuristics + plugin/built‑in). Each candidate is validated in a shadow copy against failing tests; the first green candidate is promoted. Configure external proposer via `NERION_PROPOSER_PY` or `NERION_PROPOSER_CMD`.
  - Bench runner: parallel candidate evaluation (set `NERION_BENCH_PAR` to workers), flaky retry for subset (`NERION_BENCH_FLAKY_RETRY=1`), and automatic shadow GC between candidates.
    - Flaky quarantine (full run): rerun failing nodeids individually; treat as pass if each fails-but-passes on rerun.
      - Env: `NERION_BENCH_QUARANTINE_RERUNS` (default 1), `NERION_BENCH_QUARANTINE_REQ_OK` (default 1)
    - Resource caps for parallel candidate scoring (POSIX best‑effort):
      - `NERION_BENCH_RLIMIT_AS_MB` (address space), `NERION_BENCH_RLIMIT_NOFILE`, `NERION_BENCH_RLIMIT_NPROC`, `NERION_BENCH_NICE`
    - Coverage snapshot: `NERION_BENCH_COV=1`; warn on drops beyond threshold with `NERION_BENCH_COV_WARN_DROP` (percentage points)
  - Bench coverage (optional): set `NERION_BENCH_COV=1` to run a coverage snapshot inside the task after a green candidate; writes `coverage.json` and compares against `coverage_baseline.json` in the task’s bench folder. Set `NERION_BENCH_COV_SAVE=1` to save/refresh the baseline.
  - Bench env knobs: `NERION_BENCH_MAX_CANDIDATES`, `NERION_BENCH_PYTEST_TIMEOUT`, `NERION_BENCH_PAR`, `NERION_BENCH_FLAKY_RETRY`, `NERION_BENCH_COV`, `NERION_BENCH_COV_SAVE`.
  - Context pack includes top suspect code windows and failure summary to guide proposers.
- Symbol graph utilities: defs/uses + import graph power better refactor previews and test impact mapping.
- JS/TS graph: `nerion graph affected --js --symbol Foo --depth 2` uses the JS/TS index (respects tsconfig paths/baseUrl) and prints a risk radius metric.
- Offline STT profiles (whisper|vosk|sphinx) with hot‑switch via `/stt <backend> [model]` (e.g., `/stt whisper small`).
- STT latency A/B: per‑turn timings logged to `out/voice/latency.jsonl`; summarize via `nerion voice metrics`.
- Web evidence quality: prefer recent sources, dedupe mirrors (m./amp.), skip hard paywalls (wsj/ft/bloomberg), extract numeric claims, flag contradictions.
- Learned bias: planner prompt includes local tool success rates to guide ambiguous choices.
- Test impact guidance (opt‑in): `NERION_TEST_IMPACT=1` runs impacted tests first (with import‑graph assistance to include tests that import modified modules). `NERION_COV_CONTEXT=1` augments impact mapping using coverage contexts (best‑effort).
- Tester: `NERION_TESTER=1` adds edge‑case stubs; `NERION_TESTER_HYPO=1` adds Hypothesis property stubs (if installed).
- Pre‑apply Reviewer: previews edits and prints security/style hints; `NERION_REVIEW_STRICT=1` blocks applies on security findings; optional external gates via `NERION_REVIEW_RUFF=1`, `NERION_REVIEW_PYDOCSTYLE=1`, `NERION_REVIEW_MYPY=1`; thresholds via `NERION_REVIEW_STYLE_MAX`, `NERION_REVIEW_RUFF_MAX`, `NERION_REVIEW_PYDOCSTYLE_MAX`, `NERION_REVIEW_MYPY_MAX`. In `NERION_POLICY=fast`, external linters are skipped for speed.
  - Reviewer applies to both AST edits and unified‑diff actions. Set `NERION_REVIEW_STYLE_MAX=0` to block any edits that introduce style hints; increase the limit for benchmark runs.
- Artifacts CLI: `nerion artifacts list` / `nerion artifacts show --path …` (pretty preview; `--copy-citations`).
- Artifacts export: `nerion artifacts export --topic "search:..." --out report.md` writes a clean Markdown digest with citations.
- Packaging helpers: `nerion package scaffold` writes a Homebrew formula and minimal macOS app runner to `out/package/`.
- Continuous learning: `nerion learn review` updates `out/learning/prefs.json` with tool success rates; `nerion learn show` prints prefs.
- Tracing: `nerion trace last` summarizes recent tool timings from the experience log.

### Repair/Bench Artifacts
- Triage and loop artifacts are saved under `out/bench/<task_name>/`:
  - `triage.json` — failing tests and first traceback frames
  - `suspects.json` — ranked suspect files with scores
  - `context.json` — compact code windows for top suspects (for prompt building)
  - `diff_*.patch` — diffs proposed during iterations (if a proposer plugin is active)

### Learning (Wisdom Engine)
- `nerion learn review`: reads `out/experience/log.jsonl`, computes per‑tool success rates with smoothing and optional decay/window, and persists to `out/learning/prefs.json` under `tool_success_rate`.
- `nerion learn show`: prints current learned preferences.
- The Parent planner prompt injects a “LEARNED BIASES” block (top tools by success rate) to nudge tool choice toward locally successful tools. When an intent is known, the block reflects per‑intent success (“LEARNED BIASES (intent: …)”).
- Bandit shaping: `NERION_BANDIT_STRATEGY=greedy|ucb|thompson` selects weights used to annotate tools; `NERION_BANDIT_EPSILON` (default 0.05) occasionally explores; for UCB use `NERION_BANDIT_UCB_C` to tune the bonus.
 - Experiment variants (recommended): use two arms such as `baseline` vs `contextual` or `baseline` vs `bandit+credit`; switching arms lets you measure the effect on success and latency locally.
 - Policy interactions: in `NERION_POLICY=safe`, learned biases apply only with sufficient separation and require more samples; in `fast`, the bias list is kept short to reduce prompt size.

### Environment Knobs
- `NERION_TEST_IMPACT=1` — run impacted tests first after apply.
- `NERION_COV_CONTEXT=1` — include coverage‑context mapping to prioritize tests that covered modified files.
- `NERION_TESTER=1` — generate edge‑case test stubs and include them in impacted run.
- `NERION_TESTER_HYPO=1` — add Hypothesis property test stubs (requires `hypothesis`).
- `NERION_REVIEW_STRICT=1` — block apply when Reviewer security gate would fail.
- `NERION_REVIEW_RUFF=1` — run Ruff on previewed changes and include issue count in review output.
- `NERION_REVIEW_PYDOCSTYLE=1` — run pydocstyle checks on previewed changes.
- `NERION_REVIEW_MYPY=1` — run mypy (ignore missing imports) on previewed changes.
- JS/TS Node and tools:
  - `NERION_JS_TS_NODE=1` — prefer AST‑precise JS/TS transforms via Node + ts‑morph (requires Node and `npm i ts-morph`).
  - `NERION_PRETTIER=0` — skip Prettier formatting in the Node runner (minimal diffs).
  - `NERION_ESLINT=1`, `NERION_TSC=1` — enable ESLint/tsc in the security gate for touched files.
  - `NERION_SEMGREP_JS=1` — opt‑in JS Semgrep findings.

### JS/TS Node Bridge & CLI
- Enable Node bridge: `export NERION_JS_TS_NODE=1` (install `ts-morph`).
- Rename (cross‑file): `nerion js rename --from Old --to New --root ./src [--dry-run]`.
- Apply actions to a file map: `nerion js apply --files files.json --actions actions.json [--write]`.
- Graph: `nerion graph affected --js --symbol Foo --depth 2 --json`.
- `NERION_REVIEW_STYLE_MAX` — max allowed style hints before blocking (e.g., `0`).
- `NERION_REVIEW_RUFF_MAX`, `NERION_REVIEW_PYDOCSTYLE_MAX`, `NERION_REVIEW_MYPY_MAX` — issue count limits for external tools.
- `NERION_STT_BACKEND=whisper|whisper.cpp|vosk|sphinx|auto` — choose offline STT backend.
- `NERION_TTS_BACKEND=pyttsx3|say|piper|coqui` — choose TTS backend (piper/coqui require local CLIs/models).
- `PIPER_MODEL_PATH=/path/to/piper-model.onnx` — model path for Piper backend.
- `COQUI_MODEL_PATH=/path/to/coqui-tts-model.pth` — model path for Coqui TTS CLI; `COQUI_TTS_CMD` to override CLI name (default `tts`).
- `NERION_STT_MODEL=tiny|small|base|…` — select Whisper/Vosk model size (requires local model install).
- `NERION_PLUGINS_REQUIRE_HASH=1` — require hash pinning for allowlisted plugins (in addition to path and symlink policy).
 - `NERION_SEMGREP=1` — enable semgrep in the unified static gate.
 - `NERION_POLICY=safe|balanced|fast` — controls static gate thresholds (safe is strict; fast skips external tools for speed).
- `NERION_BENCH_USE_LIBPYTEST=1` — run benchmark task tests in‑process (pytest.main) to avoid subprocess environment issues.
 - `NERION_BENCH_MAX_CANDIDATES` — cap the number of candidate diffs evaluated per iteration (default 3).
 - `NERION_BENCH_PYTEST_TIMEOUT` — timeout in seconds for pytest subprocess fallback (subset/full runs).
- `NERION_POLICY=safe|balanced|fast` — policy profiles: safe blocks style hints and security; fast reduces timeouts/concurrency and skips external linters.
- `NERION_LEARN_WINDOW_N` — number of recent log lines to consider (default 2000).
- `NERION_LEARN_DECAY_HALF_LIFE_DAYS` — exponential decay half‑life in days (optional).
- `NERION_LEARN_MIN_SAMPLES` — minimum (weighted) samples per tool to include in the bias map (default 3).
- `NERION_LEARN_ON_START` — auto-run background learning at startup (default ON; set `0` to disable).
- `NERION_LEARN_DEBOUNCE_S` — debounce seconds for event‑based background learning after actions (default 300).
- `NERION_LEARN_TOP_K` — number of top tools to include in prompt biases (default 6).
 - `NERION_INTENT_HINT` — override the current intent name for bias selection (useful in tests or when triage is ambiguous).
 - `NERION_BANDIT_STRATEGY=greedy|ucb|thompson` — strategy for learned weights per intent.
 - `NERION_BANDIT_UCB_C` — exploration constant for UCB (default 2.0).
 - `NERION_BANDIT_EPSILON` — exploration rate for epsilon‑greedy ordering (default 0.05).
 - `NERION_BANDIT_SEED` — set to an integer for deterministic bandit weighting order.
 - `NERION_LEARN_WILSON=1` — enable Wilson confidence interval overlap gating for per‑intent bias (softens bias when top candidates overlap).
 - `NERION_LEARN_PREV_STEP_WEIGHT` — credit given to non‑terminal steps (default 0.25).
 - `NERION_LEARN_COST_WEIGHT=1` — enable cost weighting by step duration.
  - `NERION_LEARN_MIN_SAMPLES_INTENT` — minimum per‑intent samples before trusting per‑intent bias (default 3).
  - `NERION_LEARN_CONFIDENCE_DELTA` — minimum p1−p2 separation to apply strong bias (default 0.10).
  - `NERION_LEARN_CI_Z` — Z value for Wilson CI (default 1.96).
 - `NERION_EXPERIMENT_NAME` / `NERION_EXPERIMENT_SPLIT` / `NERION_EXPERIMENT_ARMS` / `NERION_EXPERIMENT_ASSIGN_BY=query|session` — env‑only alternative to `learn ab` controls for A/B.

### Docs & Examples
- Cheatsheet: `docs/CHEATSHEET.md` — quick reference for CLI and flows
- Models: `docs/models.md` — API provider registry, env overrides, and examples
- V2 overview: `docs/nerion-v2-api.md` — architecture, provider matrix, migration checklist
- Voice: `docs/voice_and_hot_reload.md` — STT/VAD tuning and TTS reset tips
- Planner grammar: `docs/planner_grammar.md` — allowed actions, schema, and examples
- Proposer cookbook: `docs/proposer_cookbook.md` — how to write a repair proposer plugin
- Troubleshooting: `docs/troubleshooting.md` — common issues and fixes (JS/TS default import conflicts, Node bridge tips)

### Multi‑Language (JS/TS, experimental)
- Basic JS/TS transforms are supported when applying plans to files with extensions `.js`, `.ts`, `.tsx`:
  - `add_module_docstring` → prepends a `/** … */` file comment if none exists
  - `insert_function` → appends an `export function <name>() { /* TODO */ }` skeleton (with optional doc block)
  - `rename_symbol` → conservative token rename with word boundaries
- Powered by lightweight textual transformers (no heavy parser deps). Python files continue to use AST‑based transforms.
- Preview/apply flows (`nerion patch preview|apply-selected|preview-hunks|apply-hunks`) work across languages in the same plan.

### Packaging
- Homebrew/app scaffolds: `nerion package scaffold` writes `out/package/brew_formula.rb` and a minimal macOS app runner.
- Run bundle: `nerion package pack run` zips plan cache, artifacts, index, learning prefs, voice latency log, bench artifacts, and settings to `out/package/run_*.zip`.

### Optional Local LoRA (Incubator)
- `nerion self-learn fine-tune --data-since 30d --domain all` writes `out/self_learn/` with:
  - `dataset.jsonl`: simple examples derived from the experience log (input, intent, tools, success)
  - `train_config.json`: stub config for local LoRA/QLoRA adapters
  - `README.txt`: instructions to run your preferred local trainer
  No downloads; respects offline policy.

---

## Usage Examples

Voice commands (examples):

```text
“run a complete health scan”           → Sure — running the full health scan…
“run diagnostics”                      → Sure — running diagnostics…
“run smoke tests”                      → Sure — running smoke tests…
“switch to dev mode”                   → Shows developer options
“details for self-code”                → Detailed capability summary
“what can you do?”                     → Dynamic first‑person capability blurb
“what time is it / what’s today’s date” → Instant offline answers
“concise mode on / off”                → Toggle one‑sentence answers for general chat
“cancel”                               → Abort the current long‑running plan between steps
“read the file README.md”              → Reads a local file (repo‑jailed)
"summarize file app/nerion_chat.py”    → Summarizes a local file into concise bullets
“pin that for 7 days”                  → Pins the last fact with a TTL (auto‑expires)
“/say rate 180”                        → Sets TTS speaking rate
“/say voice Daniel”                    → Switches voice profile (if supported)
```

Planner‑routed commands (zero‑shot tools):

```text
“list plugins”                         → Names from allowlist or plugins folder
“run pytest smoke”                     → Compact test summary
“read https://example.com and summarize” → run web_read + summarize plan
```
- Profiles (config/profiles.yaml)
  - fast: minimal reviewer, no externals, shorter timeouts, higher parallelism.
  - balanced: defaults.
  - safe: strict reviewer (security+style), externals ON, longer timeouts, single-writer.
  - bench-recommended: fast + relaxed style gate + in‑process pytest + coverage assist.
  - Bench runner uses a resolver (auto) to pick bench-recommended; you can still override with `--profile`.
  - Resolver API (Phase 1): selfcoder.policy.profile_resolver decides per task; safe fallback on missing file.
  - Overrides & Safety: precedence is explicit > resolver > defaults. Sticky overrides per task live in `out/learning/prefs.json` under `profile_overrides`. Security gates are never relaxed below existing env unless you opt in.
  - Learning feedback: bench updates `profile_success[task][profile] = {ok,total,avg_latency_ms}`; resolver uses a simple bandit bias when scores tie.
  - CLI: `nerion profile show` / `nerion profile explain --task <task>` to inspect current knobs and dry-run decisions.
