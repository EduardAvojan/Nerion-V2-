Linting ignores: conventions and rationale

Purpose
- Make ignores searchable, reviewable, and self-explanatory.
- Keep exceptions rare, temporary, and documented.

General rules
- Prefer fixing the root cause over adding ignores.
- When an ignore is justified, annotate it with a short reason tag in parentheses.
- Keep reasons consistent so we can grep and audit them.

Standard format
- Inline ignores: place the reason right after the code(s).
  - Example (lazy import for perf/optional dep):
    from biglib import heavy  # noqa: E402 (lazy_import: perf)
  - Example (cycle break; TODO to refactor):
    from app.foo import Bar  # noqa: E402 (cycle: TODO split)

Allowed reason tags (initial set)
- lazy_import: perf        # defers heavy import on cold paths
- lazy_import: optional    # optional dependency; import inside function
- cycle: TODO split        # temporary until cycle is refactored out
- script: top_of_file_run  # intentional script behavior (CLI/demo)

Per-file ignores
- Prefer inline ignores for specific lines.
- For entire files that are scripts or tests with intentional top-of-file behavior, add per-file ignores in pyproject.toml with a code comment summarizing the reason.

Review cadence
- Weekly/bi-weekly audit of E402 ignores: trim legacy exceptions after refactors.
- PRs: new E402 should include a standardized reason tag and a brief code comment if non-obvious.
