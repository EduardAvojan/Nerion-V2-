# Nerion (Self-Hosted, Voice-First, Self-Improving)

Nerion is a self-hosted, privacy-first AI agent designed to run entirely offline. It combines a natural, voice-first interface with a powerful, self-evolving core. Nerion can safely analyze, refactor, and improve its own codebase, learn from new information, and automate complex tasks, all while ensuring user data remains completely private and under your control.

Designed for modularity and security, Nerion is a professional-grade platform that unites a sophisticated AI assistant with a trustworthy, autonomous software engineer.

## What is Nerion?
* **A Private Assistant:** Like Siri or Alexa, but runs 100% locally on your hardware. It manages your tasks, remembers your context, and learns your preferences without ever sending your personal data to the cloud.
* **An Autonomous Engineer:** Nerion can write, debug, and refactor its own code. You can give it high-level goals in plain English, and it will generate and safely apply the necessary code changes, complete with tests.
* **A Knowledge Engine:** Nerion can research topics on the web, read documentation, and synthesize information to answer complex questions, building a personalized knowledge base that is unique to you.

---

## Quick Start

1.  **Install in Editable Mode**:
    After creating a virtual environment, run:
    ```bash
    pip install -e .
    ```
    This makes the `nerion` and `nerion-chat` commands available in your PATH, linked directly to your source code.

2.  **Configure and Run**:
    * Set your preferences in `app/settings.yaml` (e.g., microphone, LLM model).
    * Start the agent with: `bash scripts/run_local.sh`

3.  **Interact with Nerion**:
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

### Quick Sanity Checks (LLM Automation)

- Chat model pinned by profile (fast → deepseek‑r1:14b):
  ```bash
  nerion profile explain --task chat | jq .
  # expect env includes: "NERION_LLM_MODEL": "deepseek-r1:14b"
  ```
- TS routing picks Qwen (with verbose router line):
  ```bash
  NERION_ROUTER_VERBOSE=1 nerion plan --llm -f a.ts -i "add header"
  # expect a one‑liner like: [router] task=code lang=ts model=qwen2.5-coder
  ```
- Python routing picks DeepSeek Coder V2:
  ```bash
  nerion plan --llm -f a.py -i "add docstring"
  ```
- Optional autopull (local only):
  ```bash
  export NERION_AUTOPULL=1 NERION_ALLOW_NETWORK=1
  NERION_ROUTER_VERBOSE=1 nerion plan --llm -f a.ts -i "add header"
  # router will provision a missing preferred model when allowed
  ```

### Offline Agent Improvements (New)
- Built-in repair proposer (no cloud): proposes surgical patches from failing logs (imports, NameError typos → safe rename, None/Index/Type/Value errors, numeric tolerance). Integrated shadow‑eval promotes green candidates. Used automatically when no plugin is present.
- Planner strictness + cache: `nerion plan --llm --json-grammar` enforces JSON plan output (schema-validated) and caches normalized plans in `.nerion/plan_cache.json`.
- Multiple local model backends: unified coder supports `ollama`, `llama_cpp`, `vllm`, and `exllamav2`. See [docs/models.md](docs/models.md) for setup and `nerion models bench` for quick latency.

Key env vars:
- `NERION_CODER_BACKEND` = `ollama|llama_cpp|vllm|exllamav2`
- `NERION_CODER_MODEL`, `NERION_CODER_BASE_URL`, `LLAMA_CPP_MODEL_PATH`, `EXLLAMA_MODEL_DIR`
- `NERION_JSON_GRAMMAR=1`, `NERION_LLM_STRICT=1`

> For a compact command reference, see the Cheat Sheet: `docs/CHEATSHEET.md`.

---

## Features

### Core Architecture
-   **Local-First & Private:** All core logic, memory, and sensitive data processing occurs on your device.
-   **Sovereign & Controllable:** You own the agent, the code, and the data. All major functions are controllable via CLI and configuration files.
-   **Modular & Extensible:** A secure plugin system allows for adding new tools, commands, and self-coding capabilities.
-   **Resilient by Design:** Features defensive programming and graceful degradation to ensure the agent remains operational even if optional components fail.

### Autonomous Capabilities
-   **Safe Self-Coding Engine:** Uses Abstract Syntax Tree (AST) editing for precise and syntactically correct code modifications, not fragile text replacement.
-   **Safety & Rollback Guard:** Automatically snapshots the codebase before any change. If a post-change healthcheck or test fails, it automatically reverts, ensuring the agent never permanently breaks itself.
-   **Proactive Self-Improvement:** Can be configured to periodically scan its own codebase for "code smells," vulnerabilities, or areas for improvement, and will autonomously generate plans to upgrade itself.
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

### LLM Automation & Provisioning (New)
- Built‑in repair proposer (offline): proposes surgical patches from failing logs (imports, NameError typos → safe rename, None/Index/Type/Value errors, numeric tolerance). Integrated shadow‑eval promotes green candidates. See tests under `selfcoder/tests/`. Used automatically when no `plugins/repair_diff.py` is present.
- Planner strict JSON + cache: `nerion plan --llm --json-grammar` enforces JSON plan output (schema‑validated) and caches normalized plans in `.nerion/plan_cache.json`.
- Multi‑backend coder: unified local backends — `ollama`, `llama_cpp`, `vllm`, `exllamav2`.
- Task‑aware auto‑select: orchestrator (user mode) auto‑selects a local model family based on the prompt and ensures availability with consent prompts.
- Consent flows:
  - Ollama: auto‑pulls model after consent (`NERION_ALLOW_NETWORK=1`).
  - llama.cpp: uses `config/model_catalog.yaml` to auto‑fill GGUF URL/path and downloads after consent.
  - vLLM/exllamav2: prints exact local commands/paths. Optional autostart/prepare with `NERION_AUTOSTART_VLLM=1`, `NERION_AUTOSTART_EXL=1`.
- CLI helpers: `nerion models bench` (latency table), `nerion models ensure --auto` (select + provision). See [docs/models.md](docs/models.md).

#### Task‑Aware Router & Profiles (Updated)
- Router picks coder by file type; prefers installed models, falls back deterministically:
  - TS/JS (`.ts/.tsx/.js/.jsx/.mjs/.cjs`) → `qwen2.5-coder` → `deepseek-coder-v2` → `starcoder2` → `codellama`
  - Python/general → `deepseek-coder-v2` → `qwen2.5-coder` → `starcoder2` → `codellama`
- Integration points:
  - `plan --llm` (before planner), `chat` (before chat chain)
  - `patch` (preview/apply/tui) routes by plan `target_file`
  - `rename`, `batch` routes by first file/root
- Profiles map to env (scoped):
  - `llm.chat.model` → `NERION_LLM_MODEL` (default `deepseek-r1:14b`)
  - `llm.coder.{backend,model,base_url}` → `NERION_CODER_*`
  - `net.allow: true` → `NERION_ALLOW_NETWORK=1`
  - Inspect: `nerion profile show` / `nerion profile explain --task chat|code`
- Quality gates for TS/JS (best‑effort): planner adds `eslint_clean`/`tsc_ok`, orchestrator enforces when tools exist.
- Router tools:
  - Explain decision: `nerion models router --task code -f src/app.ts -i "add header"`
  - Bench with language filter: `nerion models bench --language ts`
  - A/B harness: `nerion models ab --tasks ts py --retries 1` (ranked by success/time)
- Telemetry & knobs:
  - Logs: set `NERION_ROUTER_LOG=1` → `.nerion/router_log.jsonl`
  - Verbose: set `NERION_ROUTER_VERBOSE=1`
  - Autopull: set `NERION_AUTOPULL=1` (respects `NERION_ALLOW_NETWORK`)

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
- Models: `docs/models.md` — local backends and provision helpers
- Voice: `docs/voice_and_hot_reload.md` — offline STT/VAD and TTS reset tips
- Planner grammar: `docs/planner_grammar.md` — allowed actions, schema, and examples
- Proposer cookbook: `docs/proposer_cookbook.md` — how to write a repair proposer plugin
- Troubleshooting: `docs/troubleshooting.md` — common issues and fixes (JS/TS default import conflicts, Node bridge tips)
- Ollama autostart: `docs/ollama_launchagent.md` — keep the local LLM daemon running on macOS via LaunchAgent

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
