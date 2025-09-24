# Grand Experiment Plan – Phase 3.5

## Goal
Run Nerion in the Phase 3 "gym" for thousands of episodes to observe emergent general intelligence traits, track surprise minimization, and evaluate improvement over time.

## Experiment Setup
- **Agent:** `phase2_scaling/agent_v2.py` extended with replay-driven learning.
- **Environment Sources:**
  - Static Phase 2 universe (`logic_v2.py`).
  - Generated tasks from `phase3_scaling/generation_worker.py` using curriculum sampling.
- **Duration:** ≥ 5,000 episodes split across batches (e.g., 50 batches × 100 episodes).
- **Iteration Flow:**
  1. Generate fresh tasks via queue/orchestrator with curriculum enabled.
  2. Run agent episodes against each task.
  3. Log outcomes, surprise, and timings via replay/telemetry.
  4. Periodically retrain the GNN via `replay_training_loop.py`.

## Metrics to Track
- Episode success/failure ratio.
- Surprise distribution over time (mean, variance).
- Replay priority shifts and sampling diversity.
- Task coverage by template family and difficulty.
- Training loss curves per replay batch.

## Infrastructure Requirements
- Automated runner script combining task generation, agent execution, and replay training.
- Batch scheduler to execute episodes sequentially or in parallel.
- Storage budget for generated tasks, telemetry, and replay logs (estimate ≥ 2 GB per 5k episodes).
- Monitoring hooks (CLI reports + JSON exports) for interim checkpoints.

## Analysis Plan
- After each batch, produce summary reports (using `metrics_report.py`).
- Plot surprise trendlines and success rates across batches.
- Identify regressions: increased surprise, declining success, or stagnation.
- Collect qualitative notes on tasks where surprise remains high to inform future scaffold refinements.

## Next Steps
1. Build the automation harness to execute batch runs (task generation + agent episodes + replay training).
2. Define data retention policy (rotate old tasks, archive key logs).
3. Draft analysis notebooks/dashboards for post-run evaluation (plots, tables).
4. Run pilot (e.g., 200 episodes) to validate instrumentation before scaling to full experiment.
