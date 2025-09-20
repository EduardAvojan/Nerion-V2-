# IDE Bridge & Patch TUI

Nerion provides two local-first tools for safe patch review and surgical apply:

1) Patch TUI (terminal)
2) Lightweight HTTP bridge for IDEs

## Patch TUI

Review and apply only selected hunks from a plan without touching other changes.

Common commands:
- `nerion patch preview plan.json` — print a unified diff per file.
- `nerion patch preview-hunks plan.json --file path/to/file.py` — show hunk indexes.
- `nerion patch apply-hunks plan.json --file path/to/file.py --hunk 0 --hunk 2` — apply selected hunks.
- `nerion patch tui plan.json` — interactive TUI for selection and gating.

TUI keys:
- SPACE: toggle current hunk
- g: run security gate on current selection (risk score + top findings)
- t: run pytest subset in a shadow (quick validation)
- ] / [: cycle files with findings in the security overlay
- J: JS/TS-only overlay; E: ESLint-only; T: tsc-only
- a: apply if gate passes and subset is green (auto-runs subset if needed)
- A: force apply

## HTTP Bridge (IDE integration)

Start the local server:
```bash
nerion serve --host 127.0.0.1 --port 8765
```

Endpoints:
- `GET /version` → `{ "version": "..." }`
- `POST /patch/preview` — body is a plan object `{ actions: [...], target_file|files }`. Returns `{ diffs: {path: unified_diff} }`.
- `POST /patch/apply` — same body; applies plan to selected files, returns `{ applied: [paths...] }`.

Tips for an editor integration:
- Generate a plan (CLI or API), call `/patch/preview` to fetch diffs and show them in an editor view.
- Let the user pick files or hunks (map to the TUI hunk indices). For full control, precompute opcodes client-side or shell out to `nerion patch preview-hunks`.
- When confirmed, call `/patch/apply` for the selected subset.

References:
- CLI implementation: `selfcoder/cli_ext/patch.py`
- HTTP server: `selfcoder/cli_ext/serve.py`
