# Phase 3 Step 1 – Environment Generator Design Brief

## Overview
Phase 3 focuses on scaling Nerion's learning loop. Step 1 establishes an automated environment generator that produces a continuous stream of "toy universes"—small but diverse coding tasks with self-contained specs, implementations, and tests. The generator must support curriculum design, data logging, and reproducibility to fuel large-scale training runs.

## Objectives
- Produce hundreds of distinct code/test bundles spanning multiple patterns (algorithms, data wrangling, class design, bug fixing).
- Parameterize each task so difficulty and structure can be varied programmatically.
- Emit metadata that captures provenance, parameters, and quality signals for downstream analytics.
- Integrate with Nerion's harness so tasks can be queued, executed, and archived without manual intervention.

## Functional Requirements
1. **Template Library**
   - Author reusable templates describing task families (e.g., arithmetic pipelines, string utilities, tree traversals, refactor challenges).
   - Support pluggable modules for generators that output code plus matching pytest suites.
2. **Parameter Sampler**
   - Provide deterministic seeding to reproduce tasks.
   - Allow curriculum policies (uniform, stratified by difficulty, surprise-weighted from telemetry).
3. **Artifact Builder**
   - Generate source files, tests, and optional documentation into isolated directories.
   - Validate syntactic correctness prior to release (lint, dry-run pytest).
4. **Metadata Logger**
   - Emit JSON or YAML manifest capturing template id, parameters, random seed, expected outcomes, and checksum of artifacts.
   - Store manifests in an indexed catalog (e.g., SQLite/duckdb/JSONL) for replay and analytics.
5. **Task Harness Integration**
   - Provide CLI/API to request `N` new tasks and register them with the agent's queue.
   - Expose hooks for lifecycle events (generated, claimed, solved, archived) writing back completion stats.
6. **Archival & Cleanup**
   - Implement rotation policy (e.g., keep `K` recent tasks locally, push older ones to long-term storage).
   - Ensure cleanup of temporary dirs while preserving metadata and diffs needed for training.

## Non-Functional Requirements
- **Determinism:** Every generated bundle must be reproducible from template id + seed.
- **Scalability:** Able to create hundreds of tasks per hour on a single machine; plan for horizontal scaling later.
- **Extensibility:** Easy to add new templates without changing the core engine.
- **Observability:** Emit structured logs for generation time, validation failures, and coverage per template family.

## Proposed Architecture
```
+-------------------+
| Template Catalog  |  (YAML/py specs)
+---------+---------+
          |
          v
+-------------------+      +---------------------+
| Parameter Sampler | ---> | Artifact Builders   |
+---------+---------+      +---------------------+
          |                          |
          |                          v
          |               +---------------------+
          |               | Validation Harness  |
          |               +----------+----------+
          |                          |
          v                          v
+-------------------+      +---------------------+
| Metadata Registry |<---->| Task Output Staging |
+-------------------+      +---------------------+
                                   |
                                   v
                         +-----------------------+
                         | Agent Integration API |
                         +-----------------------+
```

### Core Modules
- `templates/`: Python classes describing task families with `render(parameters)` methods.
- `sampler.py`: Orchestrates random seeds, curriculum policies, and template weighting.
- `builder.py`: Applies template, writes files, runs validation, and prepares manifests.
- `registry.py`: Handles metadata storage and lookup.
- `service.py`: CLI/REST entry points consumed by the agent scheduler.

## Task Template Concepts
1. **Algorithmic Micro-Challenges**
   - Sorting variants, numeric sequences, graph traversals.
   - Parameters: input sizes, edge cases toggles, performance constraints.
2. **Data Transformation Pipelines**
   - CSV/JSON normalization, schema mapping, filtering operations.
   - Parameters: field names, transformation steps, error conditions.
3. **Class & Interface Design**
   - Implementing protocol-compliant classes with inheritance/composition.
   - Parameters: method counts, optional mixins, dependency injection patterns.
4. **Bug-Fix Scenarios**
   - Inject reproducible bugs (off-by-one, state leakage, concurrency primitives) with failing tests.
   - Metadata should include fault type for evaluation.
5. **Refactor Tasks**
   - Provide baseline implementation plus quality targets (e.g., reduce duplication) validated via static checks or snapshot comparisons.

Each template should expose:
- `template_id`
- `default_parameters`
- `parameter_schema` (for validation)
- `render_code(params)`, `render_tests(params)`, optional `render_docs(params)`
- `post_validation_hook(project_path)` for custom checks.

### Initial MVP Template Set

1. **`alg_arithmetic_pipeline`**
   - **Description:** Generate a function that applies a sequence of arithmetic operations (scale, shift, clamp) to numeric lists.
   - **Parameter Schema (JSON Schema fragment):**
     ```json
     {
       "type": "object",
       "properties": {
         "length": {"type": "integer", "minimum": 3, "maximum": 20},
         "operations": {
           "type": "array",
           "items": {"enum": ["add", "subtract", "multiply", "divide", "clamp"]},
           "minItems": 2,
           "maxItems": 5
         },
         "allow_zero_division": {"type": "boolean", "default": false}
       },
       "required": ["length", "operations"],
       "additionalProperties": false
     }
     ```

2. **`bug_off_by_one`**
   - **Description:** Emit code with a deliberate boundary bug (range, slicing, or index check) paired with failing tests; agent must fix the defect.
   - **Parameter Schema:**
     ```json
     {
       "type": "object",
       "properties": {
         "bug_type": {"enum": ["range_exclusive", "missing_first", "missing_last"]},
         "data_shape": {"enum": ["list", "string", "matrix"]},
         "size": {"type": "integer", "minimum": 3, "maximum": 50}
       },
       "required": ["bug_type", "data_shape"],
       "additionalProperties": false
     }
     ```

3. **`refactor_duplicate_code`**
   - **Description:** Provide boilerplate with repeated logic blocks; target is to refactor into helper functions while preserving behavior.
   - **Parameter Schema:**
     ```json
     {
       "type": "object",
       "properties": {
         "duplication_count": {"type": "integer", "minimum": 2, "maximum": 4},
         "use_classes": {"type": "boolean"},
         "io_shape": {"enum": ["dict", "tuple", "dataclass"]}
       },
       "required": ["duplication_count", "io_shape"],
       "additionalProperties": false
     }
     ```

Default parameter sets should be committed for each template to support deterministic baselines and unit tests.

## Metadata & Storage
- Use JSON Lines file `task_catalog.jsonl` with fields:
  - `task_id`, `template_id`, `seed`, `parameters`, `artifacts_path`, `created_at`, `checksum`, `status`.
- Optional SQLite mirror for easier querying when volume grows.
- Store artifacts under `generated_tasks/<task_id>/` with subfolders for `src/`, `tests/`, `docs/`.
- Maintain symlink or manifest listing active tasks assigned to the agent.

## Validation Harness
- Run `python -m py_compile` on generated modules.
- Execute `pytest -q` to ensure baseline passes/fails per template expectation.
- Capture runtime metrics (duration, coverage) for metadata.
- On failure, log event and mark manifest with `status="invalid"` for debugging; optionally retain artifacts for inspection.

## Agent Integration
- Provide API `request_tasks(count, filters)` returning manifest descriptors.
- Supply environment variables or config file enabling the agent to locate staged tasks.
- Record agent feedback: success flag, time-to-solve, surprise metrics. Append to manifest for continual learning loops.

## Telemetry & Logging
- Structured logging via `logging` module producing JSON records.
- Metrics: generate rate, template distribution, validation pass rate, average task duration, agent success rate.
- Dashboards (future): integrate with lightweight tools (e.g., Prometheus/Grafana or textual summary reports).

## Risks & Mitigations
- **Template Drift:** Generated tasks may become trivial or repetitive. Mitigate with periodic template QA and telemetry-driven weighting.
- **Validation Bottlenecks:** Running pytest per task is costly. Batch tasks or use incremental caching.
- **Metadata Explosion:** Large catalogs become unwieldy. Introduce pruning rules and compress archives.
- **Agent Overload:** Too many unsolved tasks pile up. Implement queue limits and prioritization heuristics.

## Milestones
1. **Design Finalization (current):** Capture scope, architecture, and schemas.
2. **MVP Generator:** Implement 2–3 templates, deterministic sampling, manifest logging, and basic CLI.
3. **Harness Integration:** Wire tasks into agent pipeline with lifecycle tracking.
4. **Scaling & Telemetry:** Add more templates, parallel generation, and dashboards.
5. **Curriculum Policies:** Incorporate surprise-weighted sampling and automated difficulty adjustment.

## Open Questions
- Preferred storage backend for long-term artifact archiving (local FS vs. object store)?
- How should tasks encode performance goals (runtime/memory) for future benchmarks?
- Do we need multilingual templates (e.g., JS, Go) in the near term, or focus on Python until infrastructure matures?
- What thresholds define "solved" vs. "unsolved" for curriculum updates?

## Immediate Next Steps
1. Confirm template taxonomy and prioritize initial families for MVP (recommend algorithmic + bug-fix).
2. Define manifest schema in code and stand up registry module skeleton.
3. Prototype one template (`ArithmeticPipelineTemplate`) to exercise the pipeline end-to-end.
