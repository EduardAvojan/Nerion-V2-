Planner Grammar and JSON Plan Schema
===================================

Overview
--------
Nerion’s planner produces a strict JSON “plan” that enumerates small, safe
code transforms. Plans are validated against a schema and only a conservative
set of actions is allowed. You can generate plans with the heuristic planner or
an LLM planner.

- Heuristic: `nerion plan -i "add module docstring 'X'" -f path.py`
- LLM (strict JSON): `nerion plan --llm --json-grammar -i "…" -f path.py`

Strictness and Cache
--------------------
- `--json-grammar` + `--llm` enables strict JSON mode.
  - Env: `NERION_JSON_GRAMMAR=1`, `NERION_LLM_STRICT=1`.
- Plans are sanitized (unknown actions dropped) and cached at `.nerion/plan_cache.json`
  keyed by `(repo fingerprint, target_file, instruction)`.

Allowed Actions
---------------
Only the following action kinds are accepted (see `selfcoder/plans/schema.py`).

- `create_file`
- `insert_function`
- `insert_class`
- `replace_node`
- `rename_symbol`
- `append_to_file`
- `delete_lines`
- `ensure_test` (scaffold tests when enabled)
- `add_module_docstring`
- `add_function_docstring`
- `apply_unified_diff` (textual diff fallback for bench/repair)

Plan Object Shape
-----------------
Top-level keys:

- `actions`: list of action objects (required; non-empty)
- `preconditions`: optional list of strings (e.g., `file_exists:path.py`)
- `postconditions`: optional list of strings (e.g., `no_unresolved_imports`, `tests_collect`)
- `metadata`: optional object (free-form, e.g., `{source: "llm_coder_v2"}`)
- `bundle_id`: optional non-empty string (correlates a batch)

Action Object Shapes
--------------------
You may emit legacy `{action: "…"}` or new `{kind: "…", payload: {…}}` forms.
The validator normalizes both into an internal form.

Common payload keys:
- `path`: relative file path inside the repo jail (no absolute or `..` escapes)
- `content` or `doc`: string content (overall content budget ~300KB per plan)
- `lineno_start` / `lineno_end`: integer line bounds for line-based edits
- `symbol` (or `name` for the new form): function/class symbol to target

Examples
--------
1) Add a module docstring and scaffold tests
```
{
  "actions": [
    {"kind": "add_module_docstring", "payload": {"doc": "Module docs"}}
  ],
  "target_file": "src/mod.py",
  "postconditions": ["tests_collect"],
  "metadata": {"source": "heuristic"}
}
```

2) Insert a function with a docstring
```
{
  "actions": [
    {"kind": "insert_function", "payload": {"name": "greet", "doc": "Return hello"}}
  ],
  "target_file": "pkg/util.py"
}
```

3) Apply a unified diff (bench/repair)
```
{
  "actions": [
    {"kind": "apply_unified_diff", "payload": {"diff": "--- a/m.py\n+++ b/m.py\n@@\n-def add(a,b):\n-    return a+b\n+def add(a,b):\n+    return a+b+1\n"}}
  ],
  "postconditions": ["no_unresolved_imports"]
}
```

Preconditions & Postconditions
-----------------------------
- Preconditions (best‑effort guards): `file_exists:path/to/file.py`,
  `symbol_absent:NAME@path.py`.
- Postconditions (checked after apply): `no_unresolved_imports`, `tests_collect`.

Safety Notes
------------
- Paths must be relative to the repo; absolute/parent escapes are rejected.
- Actions list is capped; payload size is capped to avoid large diffs.
- All predicted changes are scanned by security/policy gates before apply.

CLI Tips
--------
- Preview: `nerion patch preview plan.json` or `nerion patch preview-hunks`.
- Apply selected files/hunks: `nerion patch apply-selected` / `apply-hunks`.
- LLM planner selection and provisioning: see `docs/models.md` and
  `selfcoder/cli_ext/models.py` for `nerion models bench|ensure`.

