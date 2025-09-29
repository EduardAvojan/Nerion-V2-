# Nerion Evolution Plan

## ✅ Completed (Steps 1–6)
1. **Expand Action Library**
   - AST transforms: try/except wrapper, function entry/exit logs, module docstrings, etc.
2. **Smarter Planning Layer**
   - NL → AST planner + `nerion plan` (dry-run/apply).
3. **Multi-File Batch Mode**
   - `nerion batch` for JSON actions across many files; dry-run + rollback.
4. **Auto-Test Generation**
   - `nerion autotest` to generate tests for a plan; optional apply & run.
5. **Auto-Rollback & Snapshots**
   - Snapshot before risky ops; healthcheck gate; restore on failure.
6. **Voice-Triggered Self-Coding**
   - Voice pipeline integrates planner/orchestrator; outside-repo safety.

## 🚀 Next (Steps 7–12)
7. **Cross‑File Refactoring** Completed
   - Symbol graph; rename across modules; import fix-ups; dead code prune.
8. **Self‑Aware Test Coverage** Completed
   - Run coverage; identify gaps linked to changed symbols; suggest tests.
9. **Change Simulation Mode** Completed
   - Shadow workspace to apply+test plans before touching real files.
10. **Proactive Self‑Improvement** completed
   - Static analysis smells; generate upgrade plans; schedule via CLI.
11. **Local Plugin System** completed
   - `plugins/` dir with entrypoints; hot‑reload for AST/CLI extensions.
12. **Richer NL→AST Plans & Test-First Scaffolding** ✅
   - Multi-action/conditional scopes; safer conflict handling; unified dry-run diffs.
   - New safe create/insert actions (`create_file`, `insert_function/class`) with paired test generation.
   - Diff preview required in dry-run; apply gated by healthcheck/coverage drop.
   - Security checks to block unsafe paths or names.
   - Paired test generation for new code actions is scaffolded before code insertion.
   - Ensures every newly created function/class/file comes with a minimal test stub, enforcing test-first discipline.

> Source of truth for ongoing work. Update after each milestone.

## 🌌 Future Horizon (Steps 13–20+)
13. **Self-Generated Roadmap (Meta-Planning)**
    - Nerion periodically analyzes its own repo and capabilities to identify missing features or optimizations.
    - Automatically drafts and updates a `roadmap.md` file with prioritized next steps.
14. **Automatic Dependency & Security Management**
    - Scan dependencies for updates, vulnerabilities, and license issues.
    - Auto-update safe packages and re-run tests; rollback on failure.
    - Integrate static security scanning tools like Bandit or Semgrep.
15. **Full-Project Refactoring Mode**
    - AST + cross-file dependency tracking to:
      - Rename variables/functions project-wide without breaking imports.
      - Split large files into modules.
      - Reorganize into cleaner architectures.
16. **Autonomous Bug Ticket Resolution**
    - Integrate with an issue tracker (local or remote).
    - Detect reproducible bugs via tests, apply fixes, and re-run tests.
    - Commit only if bug is resolved.
17. **Lint, Style, and Doc Enforcement**
    - Self-enforce PEP8 and custom style rules.
    - Generate missing docstrings and API documentation.
    - Refactor unclear code into readable, documented functions.
18. **Live Performance Profiling & Optimization**
    - Benchmark key functions.
    - Identify bottlenecks and suggest/apply optimizations.
    - Generate before/after performance reports.
19. **Autonomous Plugin Ecosystem**
    - Learn to install and integrate plugins for:
      - Extra AST transforms.
      - Specialized analysis (e.g., DB migration generators).
      - New language support.
20. **Self-Learning from Code History**
    - Analyze git commit history to see which changes succeed or fail.
    - Learn strategies that produce stable improvements.
    - Adjust future self-coding behavior accordingly.

## Phase 3 – Structural Brain Experiments (2025-09-29)
- **Datasets & Signals:** `experiments/datasets/gnn/latest/<timestamp>/dataset.pt` now packages structural logits, semantic channels, and per-edge roles. The training CLI accepts explicit dataset paths so new snapshots remain immutable while `manifest.json` tracks provenance.
- **Training Workflow:** `training/run_training.py` exposes `--pooling`, `--num-layers`, `--residual`, `--dropout`, and `--attention-heads` switches. Each epoch logs loss/accuracy plus ROC-AUC and F1, and the CLI writes run metadata + checkpoints into `experiments/runs/gnn/<timestamp>/` while updating `digital_physicist_brain.pt`/`.meta.json` for the active brain.
- **Sweeps & Versioning:** `training/sweep.py` fans out over architecture, pooling, residual flags, dropout, and head counts. Results land in structured combo directories (e.g. `experiments/runs/gnn_sweeps/archgat_h256_poolsum_res1_.../`), each with `metrics.json` capturing history and best epoch stats alongside the checkpoint.
- **Current Best Model:** Residual GAT (256 hidden channels, sum pooling, 4 heads) leads with val acc 0.725 / AUC 0.773 / F1 0.732 (run `20250929T011032Z`). The metadata file mirrors these hyperparameters so structural vetting instantiates the correct encoder + pooling when grading lessons.

## Self-Supervision & Pretraining (2025-10-12)
- **Pretraining Corpus:** `training/dataset_builder.py --mode pretrain` exports unlabeled curriculum graphs into `experiments/datasets/gnn/pretrain/<timestamp>/` with manifests mirroring provenance and feature counts.
- **Masked-Node Objective:** `training/pretrain.py` runs a masked-node reconstruction loop (configurable `--mask-prob`, architecture, residual, heads) and saves `digital_physicist_pretrain.pt` plus `digital_physicist_pretrain.meta.json` so the latest encoder warm-start is discoverable.
- **Fine-Tune Hook:** `training/run_training.py` now accepts `--pretrained` to load the masked-node weights (with graceful head-mismatch handling) before supervised training, and records the source path in the refreshed `digital_physicist_brain.meta.json`.
- **Observed Lift (2025-09-29 batch):** Cold-start GAT peaked at val acc 0.70 / AUC 0.65 / F1 0.65 (`20250929T023909Z`); warm-starting from `digital_physicist_pretrain.pt` reached 0.75 / 0.68 / 0.64 (`20250929T024306Z`) and extended sweeps pushed to 0.775 / 0.69 / 0.71 with dropout 0.3 (lr 1e-3).
- **Monitoring:** `scripts/structural_metrics_report.py` summarises pass/fail/Δ trends from `out/learning/structural_metrics.jsonl`; schedule it post-deploy to keep lesson health visible.
