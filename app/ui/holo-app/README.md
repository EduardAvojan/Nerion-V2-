# Nerion HOLO Electron Shell (Phase 0)

This package hosts the Electron-based UI shell described in the Nerion UI roadmap.
It is intentionally lightweight for Phase 0, focusing on the IPC contract and shared
layout primitives. Subsequent phases will extend the renderer with richer
explainability and tooling surfaces.

## Getting Started

```bash
cd app/ui/holo-app
npm install
npm run start
```

The shell spawns the Nerion Python runtime (defaults to `python3 -m app.nerion_chat`)
and bridges JSONL events between the processes. Configure alternative commands via
environment variables before launching:

- `NERION_PYTHON` – override Python executable (default: `python3`).
- `NERION_PY_ENTRY` – module or script (default: `app.nerion_chat`).
- `NERION_PY_ARGS` – additional arguments appended after the module.

## Layout Overview

Phase 0 implements the base gradient background, center microphone arc container,
and right-aligned thought ribbon placeholder. Styling tokens live in
`src/styles.css` under the `:root` selector.

## IPC Schema

The contract for JSONL events exchanged with the Python runtime is documented in
`ipc/schema-v1.json`. The Electron shell validates payload shapes at runtime and logs
human-friendly errors if an unknown event is observed.

## Development Notes

- `src/main.js` defines the Electron main process, tray integration, and global
  shortcuts (press-to-talk toggle at `CmdOrCtrl+Shift+Space`).
- `src/preload.js` exposes a safe bridge limited to `send` and `on` helpers for
  renderer scripts.
- `src/renderer.js` maintains a reducer-style state machine that drives DOM updates
  without an additional framework.

Later roadmap phases can safely import and extend this structure without replacing
Phase 0 foundations.
