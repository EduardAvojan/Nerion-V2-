# Production Hardening PRs (P0–P6)

This document summarizes the PRs implemented in this cycle with intent, scope, commands, and validation steps.

## PR‑P0 — Crash‑safe I/O, Rotation, Schema v3
- Intent: Uncorruptable prefs/logs, bounded growth.
- Code: `selfcoder/learning/continuous.py`, `app/logging/experience.py`, `selfcoder/learning/jsonschema/prefs_v3.json`.
- Env: `NERION_LOG_ROTATE_BYTES=52428800`.
- Validate: `nerion learn review`; see `schema_version: 3`. Trigger rotation with small threshold.

## PR‑P1 — Robust Learning (Shrinkage + ESS Cap)
- Intent: Stabilize per‑intent learning.
- Code: `selfcoder/learning/continuous.py`.
- Env: `NERION_LEARN_EFF_N_MAX=1000`, `NERION_LEARN_HYSTERESIS_M=3`, `NERION_LEARN_MIN_IMPROVEMENT_DELTA=0.02`.
- Validate: `nerion learn review`; compare per‑intent vs global; observe reduced flapping.

## PR‑P2 — A/B Decisions + Guardrails
- Intent: Eliminate peeking bias; surface breaches.
- Code: `selfcoder/learning/abseq.py`, `selfcoder/learning/guardrails.py`, aggregator/CLI updates.
- Env: `NERION_GUARDRAIL_ERR=0.10`, `NERION_GUARDRAIL_P95=8000`, `NERION_GUARDRAIL_ESC=0.15`.
- Validate: `nerion learn ab status --refresh` shows decisions and guardrails.

## PR‑P3 — Shadow Runs + Rollout Scaffolding
- Intent: Safe self‑upgrades without impacting live.
- Code: `selfcoder/upgrade/shadow.py`, engine hook.
- Env: `NERION_UPGRADE_SHADOW=1`.
- Validate: Interact; inspect `out/policies/upgrade_state.json` → `shadow` entries.

## PR‑P4 — Observability + Replay/Export
- Intent: Operator‑grade visibility and reproducibility.
- Code: `selfcoder/cli_ext/health.py`, `selfcoder/learning/report.py`, `selfcoder/cli_ext/learn.py`.
- Commands: `nerion health dashboard`, `nerion learn replay --since 30d`, `nerion learn export --window 30d`.

## PR‑P5 — Context Hook (Back‑Compatible)
- Intent: Prepare for contextual bandits.
- Code: `app/parent/driver.py`, `app/parent/prompt.py`.
- Behavior: No change today; context ignored by prompt builder.

## PR‑P6 — Privacy & Scope Tags; Merge Policy
- Intent: Avoid cross‑user/workspace bleed.
- Code: `selfcoder/learning/continuous.py`.
- Env: `NERION_SCOPE_WS`, `NERION_SCOPE_PROJECT`.
- Behavior: Learning maps merge only when `stats.scope` matches; `personalization` always merges.
