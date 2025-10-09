"""
Test utilities for orchestration.

This module provides utilities for test collection, validation,
failure analysis, and test impact prediction.
"""
import ast
import os
from pathlib import Path
from typing import Any, Dict, List
from ops.security import fs_guard


# Repository root for test discovery
REPO_ROOT = fs_guard.infer_repo_root(Path('.'))

# Optional import graph builder
try:
    from selfcoder.analysis.symbols import build_import_graph as _build_import_graph
except Exception:
    _build_import_graph = None

# Optional coverage utilities
try:
    from selfcoder import coverage_utils as _covu
except Exception:
    _covu = None


def tests_collect_ok() -> bool:
    """Check if all test files parse without SyntaxError.

    Returns:
        True if all test files parse successfully, False otherwise
    """
    # Lightweight check: all test files under common dirs parse without SyntaxError
    candidates = [REPO_ROOT / "tests", REPO_ROOT / "selfcoder" / "tests"]
    any_found = False
    for root in candidates:
        if not root.exists():
            continue
        for p in root.rglob("test_*.py"):
            any_found = True
            try:
                src = p.read_text(encoding="utf-8")
                ast.parse(src)
            except Exception:
                return False
    return True if any_found else True  # no tests found -> treat as pass


def causal_from_post_failures(posts: List[str], failures: List[str]) -> List[Dict[str, Any]]:
    """Build a structured causal chain from postcondition failures.

    For unresolved imports, extract missing module names by file; for generic tokens, record status.

    Args:
        posts: List of postcondition tokens
        failures: List of failure messages

    Returns:
        List of structured failure information
    """
    out: List[Dict[str, Any]] = []
    if not failures:
        return out
    for f in failures:
        # Expected form for unresolved imports: "path: mod1, mod2, ..."
        if ":" in f and any(tok == "no_unresolved_imports" for tok in (posts or [])):
            path, mods = f.split(":", 1)
            missing = [m.strip() for m in mods.split(",") if m.strip()]
            out.append({
                "type": "no_unresolved_imports",
                "file": path.strip(),
                "missing_modules": missing,
            })
        elif f == "tests did not collect" or any(tok == "tests_collect" for tok in (posts or [])):
            out.append({
                "type": "tests_collect",
                "status": "failed",
            })
        else:
            out.append({"type": "unknown_postcondition_failure", "detail": f})
    return out


def predict_impacted_tests(modified_files: List[Path]) -> List[Path]:
    """Return a list of likely impacted test files given modified sources.

    Heuristics:
      - match by stem in test filenames under tests/ and selfcoder/tests/
      - include direct sibling tests (same directory test_*.py) if present
      - import-graph based (tests that import modified modules)
      - coverage-context mapping (opt-in via NERION_COV_CONTEXT=1)

    Args:
        modified_files: List of modified file paths

    Returns:
        List of predicted impacted test file paths
    """
    roots = [Path('tests'), Path('selfcoder/tests')]
    roots = [r for r in roots if r.exists()]
    if not roots or not modified_files:
        return []
    stems = {Path(p).stem for p in modified_files}
    out: List[Path] = []
    seen = set()

    # Heuristic 1: match by stem
    for r in roots:
        try:
            for p in r.rglob('test_*.py'):
                name = p.name.lower()
                if any(s in name for s in stems):
                    key = p.resolve()
                    if key not in seen:
                        seen.add(key)
                        out.append(p)
        except Exception:
            continue

    # Heuristic 2: direct sibling tests
    for m in modified_files:
        try:
            for sib in m.parent.glob('test_*.py'):
                key = sib.resolve()
                if key not in seen:
                    seen.add(key)
                    out.append(sib)
        except Exception:
            pass

    # Heuristic 3: import-graph based (tests that import modified modules)
    try:
        if _build_import_graph is not None:
            for r in roots:
                ig = _build_import_graph(r)
                # Precompute candidate module paths for modified files
                mod_paths = {Path(m).resolve() for m in modified_files}
                for test_file, mods in ig.items():
                    for m in mods:
                        try:
                            cand = Path('.').resolve().joinpath(*m.split('.')).with_suffix('.py')
                        except Exception:
                            cand = None
                        if cand and cand.resolve() in mod_paths:
                            key = test_file.resolve()
                            if key not in seen:
                                seen.add(key)
                                out.append(test_file)
    except Exception:
        pass

    # Heuristic 4: coverage-context mapping (opt-in via NERION_COV_CONTEXT=1)
    try:
        _CC = (os.getenv('NERION_COV_CONTEXT') or '').strip().lower() in {'1','true','yes','on'}
    except Exception:
        _CC = False
    if _CC and _covu is not None:
        try:
            cov = _covu.run_pytest_with_coverage(pytest_args=['-q'], cov_context='test')
            # Best-effort parse of context map
            ctx_map = {}
            files_meta = (cov.get('files') or {}) if isinstance(cov, dict) else {}
            for fname, meta in files_meta.items():
                ctxs = meta.get('contexts') if isinstance(meta, dict) else None
                if isinstance(ctxs, dict):
                    for ctx_id, lines in ctxs.items():
                        if not isinstance(ctx_id, str):
                            continue
                        ctx_map.setdefault(ctx_id, set()).add(fname)
            if ctx_map:
                mod_set = {str(Path(m).resolve()) for m in modified_files}
                test_files = set()
                for ctx_id, files in ctx_map.items():
                    try:
                        tfile = ctx_id.split('::', 1)[0]
                    except Exception:
                        tfile = ctx_id
                    # Normalize file paths for comparison
                    for f in list(files):
                        try:
                            p = Path(f).resolve()
                        except Exception:
                            continue
                        if str(p) in mod_set:
                            tf = Path(tfile)
                            if tf.exists() and tf.name.startswith('test_'):
                                test_files.add(tf)
                for tf in sorted(test_files):
                    key = tf.resolve()
                    if key not in seen:
                        seen.add(key)
                        out.append(tf)
        except Exception:
            pass
    return out
