# Roadmap to Implement Nerion's Futuristic UI/UX (Based on Locked Mockup 1)

This roadmap provides a phased plan to bring the updated Mockup 1 design (dark gradient background, glowing thought process orbs, confidence halo, adaptive states) into **Nerion**, starting with Electron while leaving room for future native migration.

---

## Phase 0 — Foundation & Architecture
**Duration:** Week 1 · **Status:** ✅ 100% complete

- **Define IPC Schema v1**
  - Outbound (Python → UI): `state`, `thought_step`, `confidence`, `progress`, `action_log`, `artifact_list`, `status_chip`, `speak_start/stop`, `word`.
  - Inbound (UI → Python): `ptt`, `chat`, `override`, `net`, `patch`, `open_artifact`.
- **Electron Shell**
  - Create main process: spawn Python, line-buffer JSONL bridge, tray, global hotkeys.
  - Renderer uses a single event reducer (state-driven UI).
- **Design Tokens**
  - Colors: dark → teal gradients, glowing accents.
  - Typography: Inter/SF/Segoe.
  - Motion: subtle easings (160–220ms).
- **Theming & Layout**
  - Base gradient background.
  - Responsive container for center mic arc + right-side Thought Ribbon.

---

## Phase 1 — Voice Interaction Core
**Duration:** Weeks 2–3 · **Status:** ✅ 100% complete

- **Standby & Listening States**
  - Idle “glimmer pulse” around screen edges.
  - On PTT press: glowing mic arc + “Listening…” text.
  - Transcript chip drops into flow on release.
- **Basic Conversation Timeline**
  - Minimal floating cards for user → agent exchanges.
- **Confidence Halo v1**
  - Simple percentage ring attached to result cards.
- **Audio Hooks**
  - Wire `voice_io.safe_speak/cancel` events to UI animations.

---

## Phase 2 — Thought Ribbon & Explainability
**Duration:** Weeks 4–5 · **Status:** ✅ 100% complete

- **Thought Ribbon (Right Edge)**
  - Vertical glowing orbs connected by fiber-optic line.
  - Active node pulses; completed nodes fade subtly.
- **Expandable Nodes**
  - Hover/click expands to micro-card with one-line rationale.
  - “Details” expands panel with full thought log.
- **Confidence Halo v2**
  - Add drivers (e.g., “tests matched”, “AST safe”) on hover.

---

## Phase 3 — Patch Review & Action Log
**Duration:** Weeks 6–7 · **Status:** ✅ 100% complete

- **Patch Review Screen**
  - Dual-pane diff view; hunk toggles; keyboard parity with CLI.
  - Gate overlay with grouped findings, risk summary.
  - “Safe Apply” + “Undo” affordances.
- **Action Log Timeline**
  - Transparent cards showing recent applied actions.
  - Inline progress animations (subset tests, apply hunks).

---

## Phase 4 — Artifacts & Research
**Duration:** Week 8 · **Status:** ✅ 100% complete

- **Artifact Browser**
  - Grid of artifact cards (site_queries, chunks).
  - Detail view with summary, citations, and “Speak summary.”
- **Inline Data Visualizations**
  - Minimal sparklines or live charts embedded in result cards.

---

## Phase 5 — Health & Settings
**Duration:** Week 9 · **Status:** ✅ 100% complete

- **Health Dashboard**
  - Tiles for voice stack, network gate activity, coverage, errors.
  - “Run healthcheck” streams logs directly into panel.
- **Settings Screen**
  - Voice backend, rate, device selector.
  - PTT hotkey capture.
  - Privacy defaults (offline by default).

---

## Phase 6 — Polish & Portability
**Duration:** Week 10 · **Status:** ✅ 100% complete

- **Visual Polish**
  - Glassmorphism for floating cards.
  - Adaptive gradients based on task type (analysis, creative, ops).
  - Micro-interactions (ripples, breathing animations).
- **Accessibility & Performance**
  - Full keyboard navigation.
  - Live regions for screen readers.
  - Cold start <2s; PTT → Listening <80ms.
- **Portability Prep**
  - Maintain `schemas/*.json` and golden IPC fixtures.
  - Document event-driven reducer for reimplementation in native OS clients.

---

# Deliverables by Phase
- **Phase 0:** IPC schema doc, Electron shell, tokens.
- **Phase 1:** Listening/standby visuals, transcript chips, confidence halo v1.
- **Phase 2:** Thought Ribbon, expanded explainability, halo v2.
- **Phase 3:** Patch Review UI, action log timeline.
- **Phase 4:** Artifact browser with inline charts.
- **Phase 5:** Health dashboard + settings.
- **Phase 6:** Polished visuals, accessibility compliance, native-ready architecture.

---

## Phase 2.5 — Memory Surfaces (Session + Long‑term)
**Duration:** overlaps Weeks 4–5 · **Status:** ✅ 100% complete

- **Session Memory Chips (right context panel)**
  - Recent facts Nerion promoted from the current session.
  - Actions: **Pin**, **Unpin**, **Forget**, **Edit** (inline text correction).
  - Hover shows provenance (turn id, confidence, last used time).
- **Memory Drawer (full view)**
  - Filter by scope: `session | long | pinned | expired soon`.
  - Searchable list with tags; bulk operations.
  - Metrics: promotions/decay over time.
- **IPC**
  - Out: `memory_update {scope, fact, confidence, last_used, action}`.
  - In: `memory_cmd {action: pin|unpin|forget|edit, fact_id, value?}`.
- **Backends**
  - `app/chat/memory_bridge.py` (LongTermMemory), `app/chat/memory_session.py` (SessionCache).

---

## Phase 3.5 — Self‑Coding Studio (Upgrade/Repair)
**Duration:** overlaps Weeks 6–7 · **Status:** ✅ 100% complete

- **Upgrade Lane** (agentic): shows when Nerion proposes an upgrade via `upgrade_agent`.
  - Cards: **“Plan available”**, with summary, affected files count, risk, est. time.
  - Actions: **Preview**, **Safe Apply**, **Defer**, **Dismiss**.
- **Plan Viewer**
  - High‑level steps (from `selfcoder.planner.llm_planner` or heuristics) displayed as a collapsible list.
  - Each step links to **Patch Review** with pre‑selected hunks.
- **Simulation / Shadow‑eval**
  - Inline results for subset tests per candidate (progress lane + pass/fail chips).
- **Rollback**
  - After apply, show **Undo** with countdown.
- **IPC**
  - Out: `selfcode_plan {id, source: llm|heuristic, summary, files, actions}`; `selfcode_candidate_result {id, passed, details}`; `upgrade_offer {why, score}`.
  - In: `selfcode_cmd {action: preview|safe_apply|force_apply|defer|dismiss, plan_id, files?}`.
- **Backends**
  - `app/learning/upgrade_agent.py` (maybe_offer_upgrade, handle_choice), `selfcoder/orchestrator.py`, `selfcoder/healthcheck.py`, `selfcoder/planner/*`.

---

## Phase 4.5 — Self‑Learning Controls & Explainability
**Duration:** Week 8–9 (in parallel with Artifacts/Health) · **Status:** ✅ 100% complete

- **Learning Timeline**
  - Shows recent learned preferences/rules with source (turn, tool), confidence, and effect.
  - Actions: **Approve**, **Revert**, **Adjust scope** (user/workspace/project).
- **Preference Diffs**
  - Visual diff of before/after for prefs (e.g., `NERION_LEARN_*`, router decisions, profile selection).
- **Knobs & Safeguards**
  - Toggle debounced learning, window/decay values, promotion thresholds.
- **IPC**
  - Out: `learning_event {kind, key, value, confidence, source}`.
  - In: `learning_cmd {action: approve|revert|set, key, value}`.
- **Backends**
  - `app/learning/*`, environment knobs (e.g., `NERION_LEARN_ON_EVENT`, `NERION_LEARN_DECAY_HALF_LIFE`), `selfcoder/llm_router.apply_router_env` (for routing learning).

---

# Capability Coverage Matrix (ensures nothing is left out)

| Capability | Backend refs (indicative) | UI surface | Phase |
|---|---|---|---|
| Voice PTT + TTS/STT | `app/chat/ptt.py`, `app/chat/voice_io.py`, `app/chat/tts_router.py` | PTT arc, Listening/Speaking states, transcript chips | 1 |
| Chat engine & intents (local) | `app/chat/engine.py`, `app/chat/intents.py`, `config/intents.yaml` | Conversation timeline, intent chips, adaptive background | 1–2 |
| Network gate & prefs | `app/chat/net_access.py`, `ops/security/net_gate.py` | Status chip, Grant/Revoke, Always-allow per task/domain | 1–2 |
| Web research: site-query / search | `app/chat/routes_web.py`, `selfcoder/analysis/*` | Research cards, artifacts links, citations, Speak summary | 4 |
| Patch review (diff/hunks) | `selfcoder/cli_ext/patch.py`, `selfcoder/orchestrator.py` | Dual diff, hunk toggles, Safe Apply / Undo | 3 |
| Security gate (risk) | `ops/security/*`, gate in patch CLI | Gate overlay, rule badges, risk meter | 3 |
| Artifacts (site_queries/chunks) | `selfcoder/analysis/docs`, `selfcoder/cli_ext/artifacts.py` | Artifact grid + detail, copy citations | 4 |
| Health & diagnostics | `selfcoder/healthcheck.py`, `app/chat/offline_tools.py` | Health tiles, run healthcheck stream | 5 |
| Settings (voice/hotkeys/privacy) | `core/ui/prefs.py`, env knobs | Settings screen (voice, hotkeys, offline defaults) | 5 |
| **Memory (session/long-term)** | `app/chat/memory_session.py`, `app/chat/memory_bridge.py` | Session chips, Memory drawer (pin/unpin/forget/edit) | **2.5** |
| **Self‑coding / Upgrade agent** | `app/learning/upgrade_agent.py`, `selfcoder/planner/*` | Upgrade lane, Plan viewer, Shadow-eval, Safe Apply | **3.5** |
| **Self‑learning (auto-preferences)** | `app/learning/*`, router env | Learning timeline, prefs diff, approve/revert/scope | **4.5** |
| Observability / Trace | `app/logging/experience.py`, `nerion trace last` | Recent actions & errors panel, log tail | 3–5 |
| Lint / Doctor | `nerion lint`, `nerion doctor` | Developer utilities panel with streaming output | 5 |
| Policy DSL & allowlist | `config/policy.yaml`, `plugins/allowlist.json`, `nerion policy` | Policy center: show merged policy, audit dry-runs | 3–5 |
| Plugins verify | `nerion plugins verify` | Verify modal + results view | 5 |
| IDE Bridge (HTTP) | `selfcoder/cli_ext/serve.py` | IDE Bridge status card + endpoints info | 3 |
| Models & router hints | `selfcoder/llm_router.py`, `capabilities.py` | Profile/router hint chips; model status in Health | 5 |
| Graph & rename (impacted) | `nerion graph affected`, `rename` | Impact explorer panel; rename preview | 3 |
| JS/TS helpers | `selfcoder/actions/js_ts.py`, Node bridge | JS/TS overlay (affected importers; J/E/T filters) | 3 |
| Bench repair loop | `nerion bench repair` | Bench Lab card (optional/advanced) | 4 |
| Voice metrics | `nerion voice metrics` | Voice latency mini-chart in Health | 5 |
| Version/build tag | `app/version.py` | About dialog / titlebar tag | 1 |

> **Note:** Rows in **bold** are the areas you asked to confirm (memory, self‑learning, self‑coding). They are included with explicit IPC and surfaces.

---

# Gaps found & how we cover them
1) **Memory visibility & control** was not explicit initially → added **Phase 2.5** with chips + drawer and IPC (`memory_update`, `memory_cmd`).
2) **Self‑Coding/Upgrade** UI did not exist → added **Phase 3.5** (Upgrade lane, Plan viewer, shadow‑eval, rollback, IPC).
3) **Self‑Learning explainability** missing → added **Phase 4.5** (learning timeline, prefs diffs, approve/revert/scope, knobs).
4) **Policy/Allowlist visibility** → surface in **Policy Center** under Dev utilities.
5) **Observability** → consolidated Recent Actions/Errors panel and Log tail.

All other CLI features have a mapped surface (Health, Tools, Artifacts, Graph, JS/TS, Voice metrics, IDE Bridge, etc.).

---

# End Goal
By Week 10, Nerion has a **futuristic, elegant, invisible UI** with **all core and advanced capabilities**:
- Adaptive gradients and glimmer pulses.
- Thought Ribbon explainability with confidence halos.
- Conversational, multimodal voice‑first design.
- **Self‑Coding Studio** with plan preview, shadow‑eval, safe apply/rollback.
- **Memory** management (session + long‑term) with pin/unpin/forget/edit.
- **Self‑Learning** timeline with operator controls and preference diffs.
- Patch Review & Security Gate overlays.
- Artifact browser with inline charts.
- Health dashboard and Settings.
- Observability panel, Policy center, Developer utilities, IDE Bridge status.
- Packaged Electron app, portable via stable IPC to a future native client.

By Week 10, Nerion has a **futuristic, elegant, invisible UI** with **all core and advanced capabilities**:
- Adaptive gradients and glimmer pulses.
- Thought Ribbon explainability with confidence halos.
- Conversational, multimodal voice‑first design.
- **Self‑Coding Studio** with plan preview, shadow‑eval, safe apply/rollback.
- **Memory** management (session + long‑term) with pin/unpin/forget/edit.
- **Self‑Learning** timeline with operator controls and preference diffs.
- Patch Review & Security Gate overlays.
- Artifact browser with inline charts.
- Health dashboard and Settings.
- Packaged Electron app, portable via stable IPC to a future native client.
