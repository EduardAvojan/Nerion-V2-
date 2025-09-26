# Nerion AGI Project Journal: The "Digital Physicist" Approach

**Vision:** To build an AGI agent named Nerion capable of recursive self-improvement by developing a deep, causal, and predictive model of its own codebase.

---

## Phase 1: The Foundation (The "Toy Universe")

**Goal:** Create a minimal, end-to-end working version of the architecture to prove the core concepts.

**Environment:**
- `math_logic.py`: A simple module with `add` and `multiply` functions.
- `test_math_logic.py`: A `pytest` suite for the module.

**Components:**
1.  [x] **Causal Scaffold:** An AST representation of the code.
2.  [x] **World Model:** A graph (`networkx`) built from the AST.
3.  [x] **Learning Loop:** A simple prediction model that learns from "surprise" (prediction error).

---

## Phase 2: Scaling the World Model

**Goal:** Introduce richer code structure, scope-aware transformations, and prepare for higher-fidelity world models.

**Environment:**
- `logic_v2.py`: Scope-sensitive module featuring shadowed names.
- `test_logic_v2.py`: Tests ensuring both local and global semantics remain intact.

**Components:**
1.  [x] **Stateful Transformer:** Rename locals without touching globals.
2.  [x] **Expanded Action Library:** Additional semantics-preserving edits.
3.  [x] **Enhanced World Model:** Richer graphs / CST integration.
4.  [x] **GNN Brain Architecture:** Neural predictor defined for graph inputs.
5.  [x] **Graph Feature Pipeline:** Convert world model graphs into PyG `Data` objects.
6.  [x] **Training Loop:** Fit the GNN on labelled code graph outcomes.
7.  [x] **High-Fidelity Editing:** Precise AST-guided text edits via `asttokens`.
8.  [x] **Agent Loop v2:** Integrate GNN predictions with environment actions and online learning.
9.  [x] **Curiosity-Driven Policy:** Select actions via surprise minimization.

---

## Phase 3: Curiosity-Driven Scaling

**Goal:** Scale the Digital Physicist prototype into a continuously learning coder through richer worlds, actions, and persistent cognition.

**Roadmap:**
1.  [x] **Environment Generator:** Design task templates; implement stochastic code/test synthesizer; add harness to rotate tasks and archive metadata.
2.  [x] **Action Library Expansion:** Catalogue refactor primitives; implement reusable scoped transformers; add reversible diff and safety checks.
3.  [x] **World Model Deepening:** Extend graphs with cross-file edges and semantic features; experiment with larger GNN backbones; evaluate feature ablations.
4.  [x] **Persistent Memory System:** Build episodic replay store; add surprise-weighted sampling; support continual fine-tuning checkpoints.
5.  [x] **Metrics & Infrastructure:** Instrument surprise/uncertainty telemetry; create regression and CI harness; stand up dataset/version tracking with distributed training hooks.

---

## Phase 4: Semantic Exploration & Generative Mastery

**Goal:** Fuse semantic understanding, richer exploration, and generative tooling so Nerion can tackle open-ended code synthesis while continuing to self-improve.

**Roadmap:**
1.  [x] **Curiosity Enhancements:** Introduce epsilon-greedy and visit-aware tie breakers; log exploration entropy for policy analysis.
2.  [x] **Semantic World Model:** Enrich graph nodes with LLM-derived embeddings and retrain the hybrid GNN on replay batches.
3.  [x] **Generative Action Library:** Add LLM-backed actions (e.g., implement-from-docstring) with safety gates, lint hooks, and replay tagging.
4.  [ ] **Exploration Regularizers (Phase 4.4):** Layer entropy bonuses/adaptive epsilon, expose policy metrics, and benchmark impact during scheduled generative runs.

---

### **Session Log**

**Session 1 (2025-09-23):**
* **Goal:** Defined the overall "Digital Physicist" architecture and the phased roadmap.
* **Progress:** Established the plan for Phase 1.
* **Next Step:** Create the `math_logic.py` and `test_math_logic.py` files.

**Session 2 (2025-09-23):**
* **Goal:** Implement Phase 1 Step 1 by standing up the toy universe and its tests.
* **Progress:** Added `toy_universe/math_logic.py` and `toy_universe/test_math_logic.py`; all pytest checks pass.
* **Next Step:** Build the Causal Scaffold v0.1 using Python's `ast` module.

**Session 3 (2025-09-23):**
* **Goal:** Deliver Phase 1 Step 2 by wiring the initial AST scaffold.
* **Progress:** Added `toy_universe/scaffold.py`; running it produces the AST dump for `math_logic.py` without errors.
* **Next Step:** Design the World Model v0.1 by converting the AST into a `networkx` graph.

**Session 4 (2025-09-23):**
* **Goal:** Complete Phase 1 Step 3 by translating the AST into the world model graph.
* **Progress:** Added `toy_universe/world_model.py`; graph currently captures function nodes for `add` and `multiply`.
* **Next Step:** Implement the Learning Loop v0.1 to predict test outcomes and learn from surprise.

**Session 5 (2025-09-23):**
* **Goal:** Establish the toy environment to execute actions and measure outcomes.
* **Progress:** Added `toy_universe/environment.py`; `step` runs the test suite after mutating `math_logic.py` and restores the file.
* **Next Step:** Build the Learning Loop module that leverages the environment, world model, and scaffold.

**Session 6 (2025-09-23):**
* **Goal:** Deliver the simple learning loop tying everything together.
* **Progress:** Added `toy_universe/agent.py`; running the loop shows the agent updating its success probability estimate toward zero as tests fail.
* **Next Step:** Reflect on Phase 1 outcomes and define Phase 2 objectives.

**Session 7 (2025-09-23):**
* **Goal:** Kick off Phase 2 with a scope-aware environment and transformer.
* **Progress:** Added `phase2_scaling/logic_v2.py`, `phase2_scaling/test_logic_v2.py`, and `phase2_scaling/environment_v2.py`; stateful renaming passes tests and restores files cleanly.
* **Next Step:** Expand the Phase 2 action set and integrate the transformer into the world model.

**Session 8 (2025-09-23):**
* **Goal:** Define the Phase 2 neural predictor capable of reading graph structure.
* **Progress:** Installed PyTorch Geometric dependencies and added `phase2_scaling/brain_v2.py`; the GNN instantiates successfully.
* **Next Step:** Feature-ize world-model graphs and train the GNN on action outcomes.

**Session 9 (2025-09-23):**
* **Goal:** Build the data pipeline that bridges code graphs and the GNN.
* **Progress:** Added `phase2_scaling/data_pipeline.py`; running it outputs PyG `Data` objects with node feature tensors.
* **Next Step:** Integrate action-labelled samples and begin training the GNN on predictive tasks.

**Session 10 (2025-09-23):**
* **Goal:** Train the GNN brain on labelled graph outcomes.
* **Progress:** Added `phase2_scaling/training_loop.py`; the model learns to distinguish passing vs failing graphs after augmenting node features.
* **Next Step:** Broaden the dataset with additional actions and refine features/graph structure.

**Session 11 (2025-09-23):**
* **Goal:** Upgrade editing fidelity using `asttokens` text splicing.
* **Progress:** Updated `phase2_scaling/environment_v2.py` to gather rename targets via a visitor and splice text ranges so type annotations and formatting remain intact; tests still pass post-edit.
* **Next Step:** Introduce additional scoped actions and connect predictions from the GNN-aware agent loop.

**Session 12 (2025-09-23):**
* **Goal:** Assemble the Phase 2 intelligent agent loop.
* **Progress:** Added `phase2_scaling/agent_v2.py`; the agent now perceives graph state, predicts with the GNN, executes scoped actions, stores experiences, and retrains online.
* **Next Step:** Expand the memory with diverse actions/outcomes and evaluate prediction accuracy over time.

**Session 13 (2025-09-23):**
* **Goal:** Broaden the action space and expose the agent to varied outcomes.
* **Progress:** Added `CHANGE_OPERATOR_MULTIPLY_TO_ADD` to `environment_v2.py` (deliberate failure) and updated `agent_v2.py` to sample randomly from the action set; agent now experiences both passing and failing trajectories.
* **Next Step:** Introduce additional safe actions and connect GNN predictions directly to action selection via surprise minimization.

**Session 14 (2025-09-23):**
* **Goal:** Enable curiosity-driven action selection.
* **Progress:** Replaced random action choice with an imagination loop in `agent_v2.py` that evaluates each action via the GNN, estimates uncertainty, and acts on the most informative option.
* **Next Step:** Refine hypothetical state generation so the GNN evaluates action-conditioned futures without relying on the training dataset helper.

**Session 15 (2025-09-23):**
* **Goal:** Chart the Phase 3 execution roadmap.
* **Progress:** Added the five-point scaling plan covering environments, actions, world model, memory, and infrastructure.
* **Next Step:** Begin Step 1 by drafting the environment generator design brief.

**Session 16 (2025-09-23):**
* **Goal:** Launch Phase 3 Step 1 by drafting the environment generator design brief.
* **Progress:** Created `phase3_scaling/environment_generator_brief.md` detailing objectives, architecture, templates, metadata schema, risks, and milestones for the task generator system.
* **Next Step:** Prioritize initial task template families and implement the manifest/registry skeleton for the generator MVP.

**Session 17 (2025-09-23):**
* **Goal:** Lock in initial template families and scaffold the manifest registry.
* **Progress:** Added MVP template schemas to `phase3_scaling/environment_generator_brief.md` and created `phase3_scaling/registry.py` providing a JSONL-backed manifest registry with append and query helpers.
* **Next Step:** Implement the manifest schema validation layer and scaffold the template modules for the MVP generator.

**Session 18 (2025-09-23):**
* **Goal:** Deliver Phase 3.1.A manifest validation.
* **Progress:** Added schema enforcement to `phase3_scaling/registry.py`, introduced explicit status lifecycle constants, hardened the loader against malformed entries, and created `tests/phase3_scaling/test_registry.py` with pytest coverage for happy-path and failure scenarios.
* **Next Step:** Move to Phase 3.1.B by scaffolding the initial template modules and producing baseline artifacts.

**Session 19 (2025-09-23):**
* **Goal:** Complete Phase 3.1.B template scaffolding and baselines.
* **Progress:** Added template base classes and concrete implementations for arithmetic, bug-fix, and refactor tasks (`phase3_scaling/templates/*`), generated deterministic baseline artifacts with manifests via `phase3_scaling/generate_baseline_tasks.py`, and introduced pytest coverage validating rendered code (`tests/phase3_scaling/test_templates.py`).
* **Next Step:** Advance to Phase 3.1.C by wiring a sampler/builder harness that consumes these templates and logs tasks end-to-end.

**Session 20 (2025-09-23):**
* **Goal:** Ship Phase 3.1.C generator harness MVP.
* **Progress:** Implemented weighted template sampling (`phase3_scaling/sampler.py`), a manifest-aware task builder with per-task directories (`phase3_scaling/builder.py`), and a CLI service that batches generation with configurable output roots (`phase3_scaling/service.py`). Added integration/unit coverage for sampler, builder, and CLI workflows (`tests/phase3_scaling/test_sampler.py`, `tests/phase3_scaling/test_builder.py`, `tests/phase3_scaling/test_service.py`).
* **Next Step:** Move into Phase 3.2 by layering telemetry/metrics and preparing curriculum-aware sampling policies.

**Session 21 (2025-09-23):**
* **Goal:** Deliver Phase 3.2.A telemetry instrumentation.
* **Progress:** Added structured telemetry logging (`phase3_scaling/telemetry.py`) and integrated per-task metrics plus run summaries into the builder and CLI (`phase3_scaling/builder.py`, `phase3_scaling/service.py`). Extended tests to assert telemetry emission for both unit and CLI flows (`tests/phase3_scaling/test_builder.py`, `tests/phase3_scaling/test_service.py`).
* **Next Step:** Proceed to Phase 3.2.B to implement curriculum-aware sampling policies leveraging the new telemetry signals.

**Session 22 (2025-09-23):**
* **Goal:** Complete Phase 3.2.B curriculum-driven sampling.
* **Progress:** Implemented telemetry-driven curriculum policies (`phase3_scaling/curriculum.py`), exposed the `--curriculum` flag in the generator CLI (`phase3_scaling/service.py`), and updated the builder to emit per-task telemetry used for adaptive weighting. Added unit tests for curriculum statistics and integration coverage demonstrating curriculum runs (`tests/phase3_scaling/test_curriculum.py`, `tests/phase3_scaling/test_service.py`).
* **Next Step:** Advance to Phase 3.3 by introducing persistent memory/replay integration that consumes telemetry and manifests for long-horizon learning.

**Session 23 (2025-09-23):**
* **Goal:** Deliver Phase 3.3.A persistent replay store.
* **Progress:** Added JSONL-backed replay storage with priority sampling (`phase3_scaling/memory.py`), integrated the store into the generator harness (`phase3_scaling/builder.py`, `phase3_scaling/service.py`) so experiences are logged alongside manifests, and authored tests covering persistence, updates, sampling, and CLI integration (`tests/phase3_scaling/test_memory.py`, `tests/phase3_scaling/test_builder.py`, `tests/phase3_scaling/test_service.py`).
* **Next Step:** Move to Phase 3.3.B to connect agent outcomes and surprise metrics back into the replay store for continual learning updates.

**Session 24 (2025-09-23):**
* **Goal:** Complete Phase 3.3.B outcome integration.
* **Progress:** Added an outcome logging utility to update replay entries with status and surprise (`phase3_scaling/outcomes.py`), extended the replay store with lookup helpers (`phase3_scaling/memory.py`), and introduced unit tests ensuring updates adjust priorities and emit telemetry (`tests/phase3_scaling/test_outcomes.py`, updated replay tests).
* **Next Step:** Proceed to Phase 3.3.C by wiring replay sampling directly into the agent's learning loop for continual fine-tuning.

**Session 25 (2025-09-23):**
* **Goal:** Deliver Phase 3.3.C replay-driven training hooks.
* **Progress:** Added replay sampling utilities that transform experiences into PyG graph batches (`phase3_scaling/replay_sampler.py`), introduced replay-driven training steps and CLI to fine-tune the GNN (`phase3_scaling/replay_trainer.py`, `phase3_scaling/replay_training_loop.py`), and expanded tests to exercise sampling, training, and integration (`tests/phase3_scaling/test_replay_sampling.py`). Updated builder metadata to ensure experiences reference source paths for training.
* **Next Step:** Transition to Phase 3.4 by layering metrics/dashboarding for long-horizon monitoring and preparing distributed task-generation pipelines.

**Session 26 (2025-09-23):**
* **Goal:** Begin Phase 3.4 with metrics aggregation and reporting.
* **Progress:** Built a metrics aggregator summarizing manifests, replay, and telemetry (`phase3_scaling/metrics.py`), added a CLI reporter for JSON summaries (`phase3_scaling/metrics_report.py`), and wrote tests validating aggregation and CLI output (`tests/phase3_scaling/test_metrics.py`, `tests/phase3_scaling/test_metrics_report.py`).
* **Next Step:** Continue Phase 3.4 by planning distributed task-generation orchestration and dashboard integration.

**Session 27 (2025-09-23):**
* **Goal:** Start Phase 3.4.B distributed generation scaffolding.
* **Progress:** Added a persistent queue manager and worker harness for batching generation requests (`phase3_scaling/generation_queue.py`, `phase3_scaling/generation_worker.py`), and introduced tests verifying queue persistence and worker execution (`tests/phase3_scaling/test_generation_queue.py`, `tests/phase3_scaling/test_generation_worker.py`).
* **Next Step:** Expand the distributed pipeline with concurrency controls, worker telemetry, and orchestration scripts.

**Session 28 (2025-09-23):**
* **Goal:** Complete Phase 3.4.C orchestration enhancements.
* **Progress:** Added global queue locking and worker telemetry (`phase3_scaling/generation_queue.py`, `phase3_scaling/generation_worker.py`), created an orchestration CLI to drain the queue (`phase3_scaling/generation_orchestrator.py`), and extended tests for telemetry, concurrency, and orchestrator execution (`tests/phase3_scaling/test_generation_worker.py`, `tests/phase3_scaling/test_generation_orchestrator.py`).
* **Next Step:** Move into Phase 3.5 by integrating these distributed components with dashboarding and planning resource allocation for large-scale runs.

**Session 29 (2025-09-23):**
* **Goal:** Initiate Phase 3.5 grand experiment planning.
* **Progress:** Authored the batch experiment plan outlining automation flow, metrics, infrastructure, and analysis strategy (`phase3_scaling/grand_experiment_plan.md`).
* **Next Step:** Implement the automation harness to execute generation + agent loops in large batches and validate a pilot run.

**Session 30 (2025-09-23):**
* **Goal:** Resolve the outstanding Session 14 follow-up by generating accurate hypothetical states for curiosity planning.
* **Progress:** Extended the data pipeline to build PyG graphs from source strings, added non-destructive action previews in `phase2_scaling/environment_v2.py`, and rewired `phase2_scaling/agent_v2.py` to imagine actions using these previews so the curiosity policy no longer depends on the training dataset helper.
* **Next Step:** Integrate the refined imagination loop into the Phase 3 automation harness and capture replay-ready telemetry.

**Session 31 (2025-09-23):**
* **Goal:** Build the initial automation harness that runs the preview-driven agent through experiment batches.
* **Progress:** Added a single-episode API and quiet mode to `phase2_scaling/agent_v2.py`, introduced the batch runner CLI (`phase3_scaling/experiment_runner.py`) that logs telemetry and replay outcomes, and created pytest coverage in `tests/phase3_scaling/test_experiment_runner.py`.
* **Next Step:** Extend the harness to operate on generated task manifests and trigger replay fine-tuning cycles after each batch.

**Session 32 (2025-09-24):**
* **Goal:** Finalize Phase 3.5 by integrating replay fine-tuning and persisting pilot experiment metrics.
* **Progress:** Wired batch automation to consume registry manifests and run post-batch replay training (`phase3_scaling/experiment_runner.py`, `phase3_scaling/registry.py`), expanded tests to cover the new workflow, executed a 5×20 episode pilot generating 120 experiences with mean surprise ≈0.245, and exported a summary report (`phase3_scaling/experiment_runs_summary.json`).
* **Next Step:** Kick off Phase 4.1 by adding epsilon-greedy exploration and entropy telemetry before scaling to the multi-thousand-episode campaign.

**Session 33 (2025-09-24):**
* **Goal:** Launch Phase 4 semantic modelling work.
* **Progress:** Added a pluggable semantic embedder with on-disk caching (`phase2_scaling/semantics.py`), fused embeddings into the graph pipeline (`phase2_scaling/data_pipeline.py`), and threaded the richer features through the agent, environment, and replay sampler. When `NERION_SEMANTIC_PROVIDER` points to an API provider (e.g., Google Gemini), the embedder now calls the provider through Nerion's registry; otherwise it falls back to deterministic vectors. New unit tests verify epsilon-greedy exploration and semantic feature dimensions.
* **Next Step:** Benchmark the hybrid features against the pilot dataset (with Gemini embeddings enabled) and proceed to Phase 4.3 by introducing generative actions with lint/test gates.

**Session 34 (2025-09-24):**
* **Goal:** Start Phase 4.3 by adding the LLM-backed generative action library.
* **Progress:** Introduced a generative action engine that drafts function bodies via Nerion's provider registry with deterministic fallbacks (`phase2_scaling/generative.py`), wired the new `IMPLEMENT_MULTIPLY_DOCSTRING` action into the environment with lint and semantic validation gates (`phase2_scaling/environment_v2.py`), threaded action metadata/tags through the agent and experiment harness (`phase2_scaling/agent_v2.py`, `phase3_scaling/experiment_runner.py`), and updated unit coverage for the expanded policy surface (`tests/unit/test_agent_policy.py`).
* **Next Step:** Advance Phase 4 by teaching the large-scale experiment harness to schedule these generative actions alongside replay-driven fine-tuning before tackling the exploration regularizers of Phase 4.4.

**Session 35 (2025-09-24):**
* **Goal:** Teach the batch harness to schedule generative actions alongside replay cycles.
* **Progress:** Added optional forced-action scheduling to the experiment runner (`phase3_scaling/experiment_runner.py`) with a new CLI flag `--generative-per-batch`, extended the agent to support scheduled episodes (`phase2_scaling/agent_v2.py`), and expanded tests to assert replay telemetry captures the LLM-backed episodes (`tests/unit/test_agent_policy.py`, `tests/phase3_scaling/test_experiment_runner.py`).
* **Next Step:** Move into Phase 4.4 by layering exploration regularizers (e.g., entropy bonuses, adaptive epsilon) on top of the scheduled generative runs and updating metrics to track the richer policy signals.

**Session 36 (2025-09-24):**
* **Goal:** Execute Phase 4.4 by adding exploration regularizers and telemetry hooks.
* **Progress:** Introduced entropy-weighted curiosity scoring and adaptive epsilon updates in the agent (`phase2_scaling/agent_v2.py`), exposed new CLI controls for experiments (`phase3_scaling/experiment_runner.py`), and extended replay/telemetry metadata plus tests to cover the richer policy signals (`tests/unit/test_agent_policy.py`, `tests/phase3_scaling/test_experiment_runner.py`).
* **Next Step:** Benchmark the new policy knobs in multi-batch runs, tune default coefficients, and feed the resulting surprise/entropy metrics into the long-horizon analytics pipeline.

**Session 37 (2025-09-24):**
* **Goal:** Consolidate the Digital Physicist stack into the long-term package layout.
* **Progress:** Reorganized Phase 2/Phase 3 modules into `nerion_digital_physicist/` (agent, environment, generation, infrastructure, experiments), preserved the original prototypes in `sandbox_foundations/`, relocated experiment artifacts/checkpoints, and updated tests to target the new package entry points.
* **Next Step:** Continue Phase 4 by leveraging checkpointed brains for larger-scale runs and updating remaining docs to reference the new structure.

---
