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