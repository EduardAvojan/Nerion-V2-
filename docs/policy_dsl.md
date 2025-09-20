# Policy DSL Quick Guide

Nerion enforces a local, repo‑scoped policy to keep self‑coding safe. You can define allow/deny lists for actions and paths, and set size limits for predicted edits.

Policy files (first found wins):
- `.nerion/policy.yaml`
- `config/policy.yaml`

Schema (keys optional):
```yaml
policy:
  actions:
    allow: [insert_function, add_module_docstring]
    deny: [rename_symbol]
  paths:
    allow: ["selfcoder/**", "app/**"]
    deny: ["plugins/**"]
  limits:
    max_file_bytes: 200000
    max_total_bytes: 500000
    max_files: 50
  secrets:
    block: true     # Block when secret-like patterns are detected
  network:
    block_requests: true  # Treat requests.* as high severity for the gate
```

Where it’s enforced:
- Before writes: the orchestrator consults `enforce_actions`, `enforce_paths`, and `enforce_limits`.
- Security Gate: predicted changes are scanned (AST/regex + external linters) and policy findings contribute to a block with a risk score.

Helpful commands:
- `nerion policy show` — show the effective merged policy.
- `nerion policy audit` — dry‑run path/limit checks against the repo. Use `--json` to integrate with tools.

Implementation references:
- Loader/validators: `selfcoder/security/policy.py`
- Security gate: `selfcoder/security/gate.py`
- Orchestrator integration: `selfcoder/orchestrator.py`

Notes:
- Paths accept glob patterns. Deny takes precedence over allow when both match.
- If `actions.allow` is provided, only the listed kinds are allowed; `deny` can still forbid a subset.
- Limits apply to predicted diffs (bytes per file / total / file count) before any write occurs.
- Put project‑specific policies under `.nerion/policy.yaml` to avoid shipping them with code.
