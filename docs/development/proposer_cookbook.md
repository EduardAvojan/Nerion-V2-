Repair Proposer Cookbook (Bench/Repair)
=======================================

Overview
--------
The “proposer” suggests minimal code changes (diffs) to fix a failing task
directory with tests. Nerion ships a built‑in offline proposer with common
fixers. You can add your own lightweight plugin at `plugins/repair_diff.py`.

Execution Flow
--------------
1) Triage failing tests and capture traceback/context.
2) Build a set of candidate diffs (plugin + built‑in heuristics).
3) Score candidates in shadow copies: run subset tests → promote first green.
4) Confirm by running the full suite (with flaky quarantine if enabled).

CLI
---
```
nerion bench repair --task /path/to/task [--max-iters 6]
  env: NERION_BENCH_PAR, NERION_BENCH_MAX_CANDIDATES, NERION_BENCH_PYTEST_TIMEOUT
       NERION_BENCH_FLAKY_RETRY=1, NERION_BENCH_QUARANTINE_RERUNS=1
       NERION_BENCH_COV=1, NERION_BENCH_COV_WARN_DROP=1.0
```

Plugin Interface
----------------
Create `plugins/repair_diff.py` and implement one of:

```
def propose_diff(ctx: dict) -> str:
    """Return a unified diff string or '' if no suggestion."""

def propose_diff_multi(ctx: dict) -> list[str]:
    """Return several unified diff candidates (strings)."""
```

Context (`ctx`)
---------------
Triage writes `out/bench/<task>/context.json`; Nerion also injects convenience
keys when calling your plugin:

- `failed`: list of failing nodeids
- `failures`: list of traceback summaries (files/lines/message)
- `files`: ranked suspect files with small code windows
- `_task_dir`: absolute path to the task root

Example Plugin
--------------
Disabled asserts (toy example) and “fix constant” nudge:
```
from difflib import unified_diff
from pathlib import Path

def propose_diff(ctx):
    files = ctx.get('files') or []
    if not files:
        return ''
    target = files[0].get('path')
    text = Path(target).read_text(encoding='utf-8')
    # Replace lines starting with 'assert' by 'pass' (demo only)
    new = []
    for ln in text.splitlines():
        if ln.lstrip().startswith('assert'):
            lead = ln[:len(ln)-len(ln.lstrip())]
            new.append(lead + 'pass')
        else:
            new.append(ln)
    new = '\n'.join(new) + '\n'
    name = Path(target).name
    return ''.join(unified_diff(text.splitlines(True), new.splitlines(True), fromfile=f'a/{name}', tofile=f'b/{name}'))
```

Unified Diff Tips
-----------------
- Use relative filenames in headers (e.g., `a/m.py` → `b/m.py`).
- Keep diffs minimal; avoid touching tests unless necessary.
- Ensure a trailing newline in new content.

Safety & Policy
---------------
- All candidate changes are repo‑jailed and scanned by security/policy gates.
- Avoid introducing secrets or unsafe patterns (see `selfcoder/security/rules.py`).
- Keep diffs surgical and prefer additive fixes over large rewrites.

Heuristics Inspiration
----------------------
The built‑in proposer (`selfcoder/repair/proposer.py`) includes:
- Add missing import for `ModuleNotFoundError`
- NameError alias to canonical import (`np` → `import numpy as np`)
- Safe typo rename on a NameError line (local scope)
- Guard for `NoneType` attribute access (`if x is None: return None`)
- Index bounds guard; cast guards for int()/float()
- Numeric tolerance: replace `x == y` by `math.isclose(x, y, ...)`

Testing Your Plugin
-------------------
- Start with a small failing task (a file and a test) under a tmp dir.
- Run: `nerion bench repair --task /tmp/task --max-iters 1` and iterate.
- Add basic unit tests that import your module and call `propose_diff(ctx)` with
  a minimal synthetic context.

Troubleshooting
---------------
- No proposal? Print debug info from the plugin or inspect `out/bench/<task>/`.
- Shadow runs failing? Try `NERION_BENCH_USE_LIBPYTEST=1` to run tests in‑process.
- Too slow? Reduce candidates, or set `NERION_BENCH_PAR` to a small value.

