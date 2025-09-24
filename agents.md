# Nerion Agent Architecture Plan

## Vision
Build Nerion as a "Digital Physicist" that maintains a causal, simulatable understanding of its codebase and improves itself by minimizing surprise.

## Guiding Pillars
- **World Model:** Learn causal structure, not just token statistics.
- **Causal Scaffold:** Encode innate axioms of program structure and semantics.
- **Active Inference:** Drive learning by prediction error rather than external rewards.

## Phase Roadmap

### Phase 1 – Toy Universe Prototype
- Scope: `math_logic.py`, `test_math_logic.py`.
- Deliverables: AST scaffold, `networkx` world model, simple prediction loop with surprise-based updates.

### Phase 2 – Scaled World Model
- Replace AST graph with richer parsing (e.g., tree-sitter) and graph neural networks.
- Expand action space to structured patches across multiple files.

### Phase 3 – Full Active Inference
- Implement formal surprise metrics (e.g., KL divergence) to update both world model and policy.
- Introduce curiosity-driven exploration and self-improvement cycles.

## Open Research Threads
- Discovering and validating extensible causal axioms for diverse languages.
- Designing tractable simulations that approximate program execution semantics.
- Efficient credit assignment across large world models during surprise minimization.

## Operational Notes
- Maintain session-by-session updates in `project_journal.md`.
- Treat Phase 1 artefacts as experimental; iterate quickly while logging outcomes.
