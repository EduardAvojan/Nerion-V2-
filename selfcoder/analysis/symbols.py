from __future__ import annotations
from pathlib import Path
import ast
import os
from typing import Dict, List, Iterable, Any, Optional, Tuple
import json

_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "_archive",
    "backups",
}

def _cache_path(root: Path) -> Path:
    root = Path(root)
    cache_dir = root / ".nerion"
    try:
        cache_dir.mkdir(exist_ok=True)
    except Exception:
        pass
    return cache_dir / "symbol_index.json"

def _iter_pyfiles(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name

def _sig(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return 0

def build_symbol_index(root: Path) -> Dict[str, List[Path]]:
    """
    Build a minimal symbol index of top-level classes and functions.

    Parameters
    ----------
    root : Path
        Directory to scan.

    Returns
    -------
    dict: {symbol_name: [paths...]}
    """
    root = Path(root)
    index: Dict[str, List[Path]] = {}
    for path in _iter_pyfiles(root):
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            continue

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                index.setdefault(name, []).append(path)
    return index

def find_symbol(symbol: str, root: Path) -> List[Path]:
    """
    Return all file paths under root that define the given top-level function/class.
    """
    idx = build_symbol_index(root)
    return idx.get(symbol, [])


def reverse_index(root: Path) -> Dict[Path, List[str]]:
    """
    Build a reverse index mapping file -> [symbols defined].
    """
    mapping: Dict[Path, List[str]] = {}
    root = Path(root)
    for path in _iter_pyfiles(root):
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            continue
        names: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
        if names:
            mapping[path] = names
    return mapping

def _qual_from_relative(module: Optional[str], level: int, file_path: Path, root: Path) -> Optional[str]:
    """Best-effort resolve a relative import into a dotted module name.

    Walk `level` parents from file_path's parent, then append `module` if present.
    Does not require __init__.py; this is a heuristic suitable for impact estimation.
    """
    try:
        base = file_path.parent
        for _ in range(max(0, int(level))):
            if base == root or base.parent == base:
                break
            base = base.parent
        parts: List[str] = []
        try:
            rel = base.relative_to(root)
            parts = [p for p in rel.as_posix().split('/') if p]
        except Exception:
            parts = []
        if module:
            parts.extend(str(module).split('.'))
        if not parts:
            return None
        return '.'.join(parts)
    except Exception:
        return None


def build_import_graph(root: Path) -> Dict[Path, List[str]]:
    """
    Build a simple import graph: file -> list of modules it imports.
    Only records top-level import/module names, not full resolution.
    """
    root = Path(root)
    graph: Dict[Path, List[str]] = {}
    for path in _iter_pyfiles(root):
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            continue
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                # Handle absolute and relative forms; include both module and submodule names
                mod = node.module or ""
                if getattr(node, 'level', 0):
                    qual = _qual_from_relative(mod or None, int(getattr(node, 'level', 0)), path, root)
                    if qual:
                        imports.append(qual)
                        # Also include specific submodules if available
                        for alias in getattr(node, 'names', []) or []:
                            try:
                                if getattr(alias, 'name', None):
                                    imports.append(qual + "." + str(alias.name))
                            except Exception:
                                continue
                else:
                    if mod:
                        imports.append(mod)
                        for alias in getattr(node, 'names', []) or []:
                            try:
                                if getattr(alias, 'name', None):
                                    imports.append(mod + "." + str(alias.name))
                            except Exception:
                                continue
        if imports:
            graph[path] = imports
    return graph

def _collect_defs(path: Path, include_methods: bool) -> Tuple[str, List[str], List[str]]:
    """Return (path_str, top_defs, method_defs_as_Class_dot_method)."""
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return str(path), [], []
    tops: List[str] = []
    methods: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tops.append(node.name)
        elif isinstance(node, ast.ClassDef):
            tops.append(node.name)
            if include_methods:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(f"{node.name}.{sub.name}")
    return str(path), tops, methods


def _collect_uses(path: Path, symset: set[str], method_map: Dict[str, List[str]]) -> Tuple[str, List[Tuple[str, int]]]:
    """Return (path_str, uses) where uses are (symbol, line) pairs.

    For methods, when encountering attribute '.name', attribute name is matched
    to any Class.name present in method_map[name] and all such Class.name are
    recorded as used at that line.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return str(path), []
    out: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in symset:
                out.append((node.id, getattr(node, "lineno", 0) or 0))
        elif isinstance(node, ast.Attribute):
            try:
                nm = getattr(node, 'attr', None)
                if nm and nm in method_map:
                    ln = getattr(node, 'lineno', 0) or 0
                    for qual in method_map.get(nm, []) or []:
                        out.append((qual, ln))
            except Exception:
                continue
    return str(path), out


def build_defs_uses(root: Path, use_cache: bool = True, *, include_methods: bool = False, workers: Optional[int] = None) -> Dict[str, Any]:
    """Build cross-reference indexes: definitions and usages of top-level symbols.

    Returns a dict with keys:
      - defs: {symbol: [paths]}
      - uses: {symbol: [{"file": path, "line": int}]}
      - reverse: {path: [symbols_defined]}
    Uses a coarse cache keyed by file mtimes under .nerion/symbol_index.json.
    """
    root = Path(root)
    cache_file = _cache_path(root)

    # Attempt to load cache
    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            files = [str(p) for p in _iter_pyfiles(root)]
            sigs = {f: _sig(Path(f)) for f in files}
            if cached.get("sigs") == sigs and cached.get("version") == 2 and bool(cached.get("methods")) == bool(include_methods):
                return cached.get("index", {})
        except Exception:
            pass

    # Build defs (top-level) and reverse index
    files = list(_iter_pyfiles(root))
    defs: Dict[str, List[str]] = {}
    reverse: Dict[str, List[str]] = {}
    method_name_map: Dict[str, List[str]] = {}
    # Parallel or sequential collection of defs
    do_parallel = False
    if workers is None:
        do_parallel = len(files) > 200
        try:
            import os as _os
            wc = int((_os.getenv('NERION_INDEX_WORKERS') or '0').strip() or '0')
            if wc > 0:
                workers = wc
                do_parallel = True
        except Exception:
            pass
    if do_parallel:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers or None) as ex:
                for pth, tops, methods in ex.map(lambda p: _collect_defs(p, include_methods), files):
                    if tops:
                        reverse[pth] = tops
                        for n in tops:
                            defs.setdefault(n, []).append(pth)
                    for qual in methods:
                        cls, _, m = qual.partition('.')
                        defs.setdefault(qual, []).append(pth)
                        method_name_map.setdefault(m, []).append(qual)
        except Exception:
            # Fallback sequential
            for path in files:
                pth, tops, methods = _collect_defs(path, include_methods)
                if tops:
                    reverse[pth] = tops
                    for n in tops:
                        defs.setdefault(n, []).append(pth)
                for qual in methods:
                    cls, _, m = qual.partition('.')
                    defs.setdefault(qual, []).append(pth)
                    method_name_map.setdefault(m, []).append(qual)
    else:
        for path in files:
            pth, tops, methods = _collect_defs(path, include_methods)
            if tops:
                reverse[pth] = tops
                for n in tops:
                    defs.setdefault(n, []).append(pth)
            for qual in methods:
                cls, _, m = qual.partition('.')
                defs.setdefault(qual, []).append(pth)
                method_name_map.setdefault(m, []).append(qual)

    # Prepare symbol set for usage finding
    symset = set(defs.keys())

    # Build uses: find Name loads that match known symbols
    uses: Dict[str, List[Dict[str, Any]]] = {}
    if do_parallel:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers or None) as ex:
                for pth, items in ex.map(lambda p: _collect_uses(p, symset, method_name_map), files):
                    for sym, ln in items:
                        uses.setdefault(sym, []).append({"file": pth, "line": ln})
        except Exception:
            for path in files:
                pth, items = _collect_uses(path, symset, method_name_map)
                for sym, ln in items:
                    uses.setdefault(sym, []).append({"file": pth, "line": ln})
    else:
        for path in files:
            pth, items = _collect_uses(path, symset, method_name_map)
            for sym, ln in items:
                uses.setdefault(sym, []).append({"file": pth, "line": ln})

    index = {"defs": defs, "uses": uses, "reverse": reverse}

    # Save cache
    if use_cache:
        try:
            files = [str(p) for p in files]
            sigs = {f: _sig(Path(f)) for f in files}
            blob = {"version": 2, "sigs": sigs, "index": index, "methods": bool(include_methods)}
            cache_file.write_text(json.dumps(blob), encoding="utf-8")
        except Exception:
            pass

    return index

__all__ = [
    "build_symbol_index",
    "find_symbol",
    "reverse_index",
    "build_import_graph",
    "build_defs_uses",
]
