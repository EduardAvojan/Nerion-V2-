Troubleshooting
===============

JS/TS: Default Import Conflict
------------------------------
When merging imports, the Node bridge prevents multiple default imports from the
same module in a single file. If it detects more than one default import for the
same module, it fails with an actionable message.

Example
```
// Before
import Default from 'lib';
import Foo from 'lib'; // second default import from 'lib' → conflict
```

Runner output (concise):
```
[js.ts] error default import conflict for 'lib': Default, Foo action=update_import index=0 file=.../file.ts
[js.ts] error code=IMPORT_DEFAULT_CONFLICT suggestion=Consider converting one default to named: import { Default as Alias } from 'lib'; or consolidate into a single default import.
```

Remediation options
- Convert one default import to a named specifier (with an alias if needed):
  ```
  import Default from 'lib';
  import { Foo as Foo } from 'lib';
  ```

- Or consolidate to a single default import and refactor call sites to use named imports:
  ```
  import Default from 'lib';
  // Replace usages of Foo with Default or import Foo as a named specifier
  ```

Notes
- The bridge preserves import cluster ordering and comments. It will not silently
  pick one default over another to avoid changing semantics without intent.
- If the module doesn’t actually export a named specifier for the symbol, flip
  strategies (e.g., keep a single default import and convert the other usage to
  a named import from the right module).

Node Bridge Tips
----------------
- Enable AST‑precise transforms: `export NERION_JS_TS_NODE=1` (requires Node and `npm i ts-morph`).
- Skip auto‑format (minimal diffs): `export NERION_PRETTIER=0`.
- Only enable ESLint/tsc checks when needed: `NERION_ESLINT=1`, `NERION_TSC=1`.
- JS Semgrep opt‑in: `NERION_SEMGREP_JS=1`.

