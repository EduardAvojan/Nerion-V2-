# Enable Node Bridge for JS/TS (ts‑morph)

Nerion can use a Node.js bridge powered by ts‑morph for precise JS/TS transforms (imports, symbol renames, multi‑file updates). When the bridge is disabled or missing, Nerion falls back to safe textual transforms.

## Quick Setup

1) Install Node.js (v16+ recommended).
2) From the repository root, install ts‑morph:
```bash
npm init -y            # if you don’t have a package.json yet
npm install ts-morph   # local install in this repo is recommended
```
This makes `node_modules/` available at the repo root. The runner resolves `require('ts-morph')` from `tools/js/tsmorph_runner.js`, so a repo‑local install is sufficient.

3) Enable the bridge for JS/TS actions:
```bash
export NERION_JS_TS_NODE=1
```

## Verify
- Confirm Node is available: `node -v`.
- Confirm the runner exists: `tools/js/tsmorph_runner.js`.
- Run a small rename preview: `nerion js rename --root ./src --from OldName --to NewName --dry-run`.
  - With `NERION_JS_TS_NODE=1`, Nerion attempts the Node bridge. Otherwise, it uses the textual fallback.

## Notes
- The Node runner is sandboxed to read an in‑memory file map and return updated sources; Nerion enforces repo‑jail when writing changes.
- For large TS monorepos, consider adding a `tsconfig.json` for better path resolution. The runner will respect `baseUrl` and `paths` if present.
- Troubleshooting: set `NERION_JS_TS_NODE=1` and ensure `node_modules/ts-morph` exists under the repo; the bridge prints concise errors and falls back gracefully when unavailable.

References:
- Bridge: `selfcoder/actions/js_ts_node.py`
- Text fallback: `selfcoder/actions/js_ts.py`
- Runner: `tools/js/tsmorph_runner.js`
