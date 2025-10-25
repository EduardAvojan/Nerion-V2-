# Nerion Cheat Sheet

Your quick reference for voice and chat commands. All commands work in the terminal chat (prefix with `>` for typed chat) and many via voice.

## Basics
- Hold `SPACE`: Push‑to‑Talk (records while held; stops on release).
- Toggle Speech: `CapsLock` (or `F9/F12`; in VS Code `Ctrl+F9`).
- Type chat: start a line with `>` (example: `> what day is it?`).

## Voice & Mode Controls
- `/speech on` or `/voice on`: Enable mic + TTS.
- `/speech off` or `/voice off`: Disable mic + TTS.
- `/mute on|off`: Mute/unmute TTS.
- `/say <text>`: Speak this text.
- `/say rate <int>`: Set TTS speaking rate (e.g., `/say rate 180`).
- `/say voice <name>`: Select voice profile if supported (e.g., `/say voice Daniel`).
- `/stt <backend> [model]`: Switch offline STT (whisper|whisper.cpp|vosk|sphinx|auto), e.g., `/stt whisper small`.
- `/q`: Skip this turn.
- `concise mode on|off`: Keep general answers to one sentence.
- `cancel`: Abort the current long operation (between steps).
 - List devices: `nerion voice devices`

## Quick Answers (Offline)
- `what time is it`: Current time.
- `what’s today’s date` or `what date is it`: Today’s date.
- `what is your name`: Identity (“Nerion”).

## System & Diagnostics
- `run a complete health scan` / `run health check`: Healthcheck (ack + spinner).
- `diagnose system health`: Same as healthcheck.
- `run diagnostics`: Local diagnostics (CPU/MEM; ack + spinner).
- `run smoke tests` / `run pytest smoke`: Pytest smoke (compact summary; ack + spinner).

## Developer Mode (Capabilities)
- `switch to dev mode` / `exit dev mode`: Toggle developer view.
- `show dev options` / `list developer options`: See topics.
- `details for <topic>`: Deep dive; topics include:
  - `self-code`, `self-improve`/`self-learning`, `memory`, `web-research`, `voice-text`, `security`, `planner`, `tools`, `plugins`, `tools-plugins`.

## Planner Tools (Zero‑Shot)
- `list plugins`: Names from allowlist or `plugins/` folder.
- `run pytest smoke`: Runs smoke tests via safe subprocess.
- Paste a URL or `read <URL>`: Deep read (will ask for permission if needed).
- `search <topic>`: Web search (permissioned).

## Files (Safe Local)
- `read the file README.md`: Reads a text file inside the repo jail.
- `summarize file app/nerion_chat.py`: Summarizes a local file (LLM when available; falls back to bullets).

## Web & Research
- `read https://example.com and summarize`: Site‑query + summarize.
- `search latest news on <topic>`: Web search + links (with permission prompt).

## Memory
- `remember that I like tea`: Store a preference/fact.
- `what do you remember about me`: Humanized recall (e.g., “You like …”).
- `forget <fact>`: Remove a matching memory.
- `unpin <fact>`: Demote a pinned item.
- `pin that/this`: Pin the last learned fact.
- `pin that for <N> days`: Pin last learned fact with a TTL (auto‑expires after N days).

## Self‑Coding (Voice Triggers)
- `upgrade yourself` / `edit yourself` / `modify your code` / `we need an upgrade`: Enter self‑coding mode.
- `apply the plan` / `go ahead with the update`: Apply a pending plan (after review gates).

## Observability
- `show last actions` / `list last 15 actions`: Recent tool calls with status and duration.
- `show last errors` / `list last errors`: Last few failed tool calls with tool name and error reason.
 - TUI overlay shortcuts: `g` (gate), `t` (tests), `I` (JS affected), `R` (Python rename affected), `J/E/T` (filters), `]`/`[` (cycle files).

## Network & Safety
- When a command needs the internet, Nerion asks: `allow / no / always (for this task)`. Your choice is respected and logged.

## CLI Handy Commands
- `nerion docs site-query --query "..." --augment`: Persist web evidence + chunks (RAG‑ready).
- `nerion docs search --topic "search:..." --contains "..." --limit 5`: Search local knowledge chunks.
- `nerion graph affected --symbol Foo`: Show defs/uses for a symbol.
- `nerion patch preview plan.json`: Unified diff preview of a plan.
- `nerion patch apply-selected plan.json --file a.py --file b.py`: Apply only selected files from a plan.
- `nerion patch preview-hunks plan.json --file a.py`: Show change hunks with indexes.
- `nerion patch apply-hunks plan.json --file a.py --hunk 0 --hunk 2`: Apply selected hunks only.
- `nerion patch safe-apply plan.json [--file ...] [--json]`: Reviewer → subset tests → apply if clean (blocks on risk/tests by default).
- `nerion patch preview plan.json --simple-diff`: Minimal colored diff output (compact).
- JS/TS helpers:
  - Enable Node bridge: `NERION_JS_TS_NODE=1` (Node + ts-morph)
  - `nerion js rename --from Old --to New --root ./src`
  - `nerion js apply --files files.json --actions actions.json [--write]`
  - `nerion graph affected --js --symbol Foo --depth 2 --details --json`
- JS/TS support (experimental): plans can target `.js/.ts/.tsx` with a limited set of actions:
  - `add_module_docstring` (adds `/** … */`), `insert_function` (adds an `export function` skeleton), `rename_symbol` (token rename).
- `nerion bench repair --task /path/to/task --max-iters 6`: Triage failing tests in a task dir and run a repair loop (writes artifacts under `out/bench/<task>/`).
  - Auto profile selection (bench-recommended) is used by default; override with `--profile`.
  - Proposer: set `NERION_PROPOSER_PY=module:propose_diff` or `NERION_PROPOSER_CMD="..."` to provide model-based diffs. The runner evaluates multiple candidates per iteration (heuristics + plugin) in a shadow and picks the first green.
  - Progress: shows per-iteration acks and per-candidate OK/FAIL lines, plus subset/full test results.
- `nerion rename --show-affected …`: Graph‑aware rename preview (non‑JSON preview).
- `nerion net status` / `nerion net prefs --clear`: Inspect/clear network gate state and saved prefs.
- `nerion net offline --for 30m`: Force offline mode for the given session window (s/m/h supported).
- `nerion plugins verify`: Verify plugins against allowlist, path policy, and optional hash pinning.
- `nerion lint` / `nerion lint --fix`: Run Ruff lint; optionally auto‑fix and format.
- `nerion doctor`: Check mic, TTS/STT backends, and print remedies.
- `nerion health dashboard`: Quick terminal health summary (experience, STT, coverage).
- `nerion health html --out out/health/index.html --refresh 10`: HTML dashboard with auto‑refresh.
- `nerion voice metrics --last 100`: Summarize recent STT latency by backend/model.
- `nerion artifacts list` / `nerion artifacts show --path <file.json>`: Browse and preview saved research artifacts; `--copy-citations` to copy links.
- `nerion artifacts export --topic "search:..." --out report.md`: Export a Markdown digest with citations.
- `nerion review <plan.json>`: Run local Reviewer (security/style) before applying a plan.
- `nerion learn review` / `nerion learn show`: Update and view learned preferences from recent outcomes.
- `nerion package scaffold`: Write Homebrew formula and minimal macOS app runner to `out/package/`.
- `nerion package pack run`: Zip run artifacts (plan cache, artifacts, index, learning prefs, voice latency, bench files, settings) to `out/package/run_*.zip`.
- `nerion preflight plan.json`: Run Reviewer + Tester preview (no writes) for a plan.
- `nerion trace last --last 20`: Summarize last N runs from the experience log (tool timings).
- `nerion trace digest --since 24h`: Human digest summary.
- `nerion trace export --since 24h --out out/report.md`: Write a Markdown digest for sharing.

## Test Impact & Reviewer
- Impacted tests first: set `NERION_TEST_IMPACT=1`.
- Coverage contexts (best‑effort): set `NERION_COV_CONTEXT=1` to prioritize tests that covered modified files.
- Tester expansion (edge stubs): set `NERION_TESTER=1`.
- Strict review (block on security findings): set `NERION_REVIEW_STRICT=1`.
 - Add Hypothesis stubs: set `NERION_TESTER_HYPO=1` (requires `hypothesis`).
 - Optional reviewer gates: `NERION_REVIEW_RUFF=1`, `NERION_REVIEW_PYDOCSTYLE=1`, `NERION_REVIEW_MYPY=1`.
  - Reviewer thresholds: `NERION_REVIEW_STYLE_MAX`, `NERION_REVIEW_RUFF_MAX`, `NERION_REVIEW_PYDOCSTYLE_MAX`, `NERION_REVIEW_MYPY_MAX`.

## Bench/Repair
- Profiles (config/profiles.yaml): fast | balanced | safe | bench-recommended.
- Auto resolver picks bench-recommended for repair; use `--profile` to override.
- `NERION_POLICY=fast` — recommended env for custom runs.
- `NERION_REVIEW_STYLE_MAX=9999` — relax style gate while keeping security on.
- `NERION_BENCH_USE_LIBPYTEST=1` — run task tests in‑process (no subprocess dependency on external pytest).
- Sticky overrides: `nerion profile set --task bench_repair --profile fast`; clear via `nerion profile clear --task bench_repair`.
- Inspect: `nerion profile show`; explain: `nerion profile explain --task apply_plan`.
- Optional proposer plugin: create `plugins/repair_diff.py` with `def propose_diff(context) -> str` returning a unified diff.

## Env: Offline STT
- `NERION_STT_BACKEND=whisper|vosk|sphinx|auto`
- `NERION_STT_MODEL=tiny|small|base|…`
- Hot switch at runtime: `/stt <backend> [model]`
- `nerion --version`: Print build version tag.

## Env: Plugins
- `NERION_PLUGINS_REQUIRE_HASH=1`: Require hash pinning for allowlisted plugins (path/symlink policy still applies).
 - JS/TS tools: `NERION_ESLINT=1`, `NERION_TSC=1`, `NERION_SEMGREP_JS=1` (opt-in semgrep for JS)

## Env: Policy Profiles
- `NERION_POLICY=safe|balanced|fast`:
  - safe: conservative; stronger confidence gates for per‑intent bias (more samples, larger delta); conservative concurrency/timeouts.
  - balanced: defaults.
  - fast: smaller bias deltas allowed; trims learned‑bias list length; favors exploration and speed.

## Env: Learning
- `NERION_LEARN_ON_START` — auto-run background learning at startup (default ON; set `0` to disable)
- `NERION_LEARN_WINDOW_N` — number of recent log lines to consider (default 2000)
- `NERION_LEARN_DECAY_HALF_LIFE_DAYS` — exponential decay half‑life in days (optional)
- `NERION_LEARN_MIN_SAMPLES` — minimum (weighted) samples per tool to include (default 3)
- `NERION_LEARN_DEBOUNCE_S` — debounce seconds for event‑based learning after actions (default 300)
- `NERION_LEARN_TOP_K` — number of top tools in prompt biases (default 6)
- `NERION_LEARN_PREV_STEP_WEIGHT` — credit for non‑terminal steps (default 0.25)
- `NERION_LEARN_COST_WEIGHT=1` — penalize long steps when weighting credits

## Env: Bandit
- `NERION_BANDIT_STRATEGY=greedy|ucb|thompson` — strategy for bias weights (per‑intent when available).
- `NERION_BANDIT_UCB_C` — UCB exploration constant (default 2.0).
- `NERION_BANDIT_EPSILON` — exploration rate for ordering (default 0.05).
- `NERION_INTENT_HINT` — override intent name used for per‑intent bias selection.

## Env: Confidence Gates
- `NERION_LEARN_MIN_SAMPLES_INTENT` — minimum samples for per‑intent bias (default 3)
- `NERION_LEARN_CONFIDENCE_DELTA` — p1−p2 separation threshold (default 0.10)
- `NERION_LEARN_CI_Z` — Z value for Wilson CI (default 1.96)

## CLI: Learning A/B
- `nerion learn ab start --name eval1 --split 0.5 --assign-by query --arms baseline,bandit+credit`
- `nerion learn ab status`
- `nerion learn ab stop`

## CLI: Self-Learn (LoRA)
- `nerion self-learn fine-tune --data-since 30d --domain all` — write dataset + training stub under `out/self_learn/`

## CLI: Learning Maintenance
- `nerion learn review` — recompute and persist learned prefs (updates `schema_version` automatically if missing)
- `nerion learn reset` — delete `out/learning/prefs.json`; recreated on next review with current schema
