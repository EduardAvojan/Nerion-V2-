Parent–Apprentice Architecture (DeepSeek ←→ Nerion)
===================================================

Overview
--------
Nerion (the Apprentice) delegates planning to a Parent LLM (in V2 this is typically a hosted provider such as Anthropic Claude via API), then safely executes the resulting plan using a typed, allow‑listed tool interface.

Key Components
--------------
- ParentDecision schema: pydantic model for `intent`, `plan` of `Step`s, `requires_network`, etc.
- ParentDriver: builds a system prompt and injects an exact tool manifest derived from `config/tools.yaml`.
- ParentExecutor: validates tool args (pydantic), enforces tool allowlist, preflights network once, and executes steps.
- Net policy: unified wrapper (app/chat/net_policy.py) that combines the master switch and session grants.
- Metrics: per‑tool outcome and latency recorded through the chat experience logger.

Safety
------
- Network gate: outbound requests are limited to the LLM providers you configure and require explicit grants when other tools need additional domains.
- Repo jail: All file I/O is guarded by `ops.security.fs_guard`.
- Security gate: Self‑improve and orchestrated writes are preflight‑scanned.

Hot‑Reload & Concurrency Notes
------------------------------
- TTS router exposes `reset()` to tear down threads/processes for hot‑reload scenarios.
- Chat state is injected (set_voice_state); be mindful in multi‑agent or swarm setups.

Self‑Learning
-------------
- An upgrade agent offers “Self Learning Upgrade” prompts when enough new knowledge accumulates and allows scheduling or snoozing.
- Mid‑session checks are performed with minimal overhead.

Getting Started
---------------
1. Tools manifest: extend `config/tools.yaml` (name, description, params). The Parent prompt uses this verbatim.
2. Runners: wire a callable per tool in `app/chat/parent_exec.py` (or inject your own into `ParentExecutor`).
3. Network: ensure a single prompt for network when `requires_network=true` via `_ensure_network_for('parent.plan')`.
4. Observe: Inspect metrics in the experience log; iterate tool prompts and descriptions for better planning.
