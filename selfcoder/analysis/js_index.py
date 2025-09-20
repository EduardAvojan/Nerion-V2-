from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
import json
import os
import re

_EXTS = ('.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs')


def _iter_js_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dp, dn, fn in os.walk(root):
        if any(part in {'.git', 'node_modules', 'dist', 'build', '.venv'} for part in Path(dp).parts):
            continue
        for name in fn:
            if name.endswith(_EXTS):
                out.append(Path(dp) / name)
    return out


def _sig(root: Path) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for p in _iter_js_files(root):
        try:
            m[str(p)] = int(p.stat().st_mtime_ns)
        except Exception:
            m[str(p)] = 0
    return m


_RE_IMPORT = re.compile(r"^\s*import\s+([\s\S]+?)\s+from\s+['\"]([^'\"]+)['\"]\s*;?\s*$", re.M)
_RE_EXPORT_FN = re.compile(r"^\s*export\s+function\s+([A-Za-z_$][\w$]*)\b", re.M)
_RE_EXPORT_CLASS = re.compile(r"^\s*export\s+class\s+([A-Za-z_$][\w$]*)\b", re.M)
_RE_EXPORT_CONST = re.compile(r"^\s*export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)\b", re.M)
_RE_EXPORT_NAMED = re.compile(r"^\s*export\s*\{([^}]+)\}\s*(?:from\s+['\"][^'\"]+['\"])?\s*;?\s*$", re.M)
_RE_EXPORT_NAMED_FROM = re.compile(r"^\s*export\s*\{([^}]+)\}\s*from\s+['\"]([^'\"]+)['\"]\s*;?\s*$", re.M)
_RE_EXPORT_ALL_FROM = re.compile(r"^\s*export\s*\*\s*from\s+['\"]([^'\"]+)['\"]\s*;?\s*$", re.M)


def _load_tsconfig(root: Path) -> Tuple[Optional[Path], Dict[str, List[str]]]:
    # returns (baseUrlPath, paths mapping)
    try:
        tc = root / 'tsconfig.json'
        if not tc.exists():
            return (None, {})
        data = json.loads(tc.read_text(encoding='utf-8'))
        co = data.get('compilerOptions') or {}
        base = co.get('baseUrl')
        base_path = (root / base).resolve() if isinstance(base, str) and base else None
        paths = co.get('paths') or {}
        norm: Dict[str, List[str]] = {}
        if isinstance(paths, dict):
            for pat, arr in paths.items():
                if isinstance(arr, list):
                    norm[str(pat)] = [str(x) for x in arr]
        return (base_path, norm)
    except Exception:
        return (None, {})


def _apply_paths_mapping(spec: str, base_path: Optional[Path], paths: Dict[str, List[str]]) -> Optional[str]:
    # Support simple * wildcard mapping: '@lib/*': ['src/lib/*']
    for pat, arr in (paths or {}).items():
        if '*' in pat:
            prefix = pat.split('*', 1)[0]
            suffix = pat.split('*', 1)[1]
            if spec.startswith(prefix) and spec.endswith(suffix):
                mid = spec[len(prefix):len(spec)-len(suffix) if suffix else None]
                for tgt in arr:
                    if '*' in tgt:
                        repl = tgt.replace('*', mid)
                    else:
                        repl = tgt
                    cand = str(((base_path or Path('.')) / repl).resolve())
                    return cand
        elif spec == pat:
            arr0 = (paths.get(pat) or [''])[0]
            cand = str(((base_path or Path('.')) / arr0).resolve())
            return cand
    if base_path is not None:
        # Try baseUrl + spec
        return str((base_path / spec).resolve())
    return None


def _resolve_module(spec: str, base: Path, root: Path, *, base_url: Optional[Path], paths: Dict[str, List[str]]) -> Optional[Path]:
    if not spec or not (spec.startswith('.') or spec.startswith('/')):
        # Try tsconfig paths/baseUrl mapping
        try:
            mapped = _apply_paths_mapping(spec, base_url, paths)
            if mapped:
                p = Path(mapped)
                # try as is + extensions + index
                if p.exists():
                    return p
                for ext in _EXTS:
                    if p.with_suffix(ext).exists():
                        return p.with_suffix(ext)
                for ext in _EXTS:
                    idx = p / f'index{ext}'
                    if idx.exists():
                        return idx
        except Exception:
            pass
        return None
    cand = (base.parent / spec)
    # Try exact, then with common extensions and index files
    if cand.exists():
        return cand
    for ext in _EXTS:
        if (cand.with_suffix(ext)).exists():
            return cand.with_suffix(ext)
    # index files
    for ext in _EXTS:
        p = cand / f'index{ext}'
        if p.exists():
            return p
    return None


def build(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    base_url, paths = _load_tsconfig(root)
    defs: Dict[str, List[str]] = {}
    imports: Dict[str, List[str]] = {}  # file -> [module/spec path or unresolved]
    import_symbols: Dict[str, Dict[str, Dict[str, Any]]] = {}  # file -> module -> {names: [imported], aliases: {imported: local}}
    reexports: Dict[str, List[Dict[str, Any]]] = {}
    files = _iter_js_files(root)
    for p in files:
        try:
            txt = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        # defs
        for rx in (_RE_EXPORT_FN, _RE_EXPORT_CLASS, _RE_EXPORT_CONST):
            for m in rx.finditer(txt):
                defs.setdefault(m.group(1), []).append(str(p))
        for m in _RE_EXPORT_NAMED.finditer(txt):
            names = [s.strip() for s in (m.group(1) or '').split(',') if s.strip()]
            for nm in names:
                if ' as ' in nm:
                    nm = nm.split(' as ')[1].strip()
                defs.setdefault(nm, []).append(str(p))
        # imports
        mods: List[str] = []
        for m in _RE_IMPORT.finditer(txt):
            clause = (m.group(1) or '').strip()
            mod = (m.group(2) or '').strip()
            if not mod:
                continue
            target = _resolve_module(mod, p, root, base_url=base_url, paths=paths)
            key = str(target) if target else mod
            mods.append(key)
            # Parse specifiers
            names: List[str] = []
            alias: Dict[str, str] = {}
            if clause.startswith('* as '):
                # namespace import — no symbol map
                pass
            else:
                # default import
                parts = clause.split(',')
                if parts and parts[0] and not parts[0].strip().startswith('{'):
                    names.append('default')
                    alias['default'] = parts[0].trim() if hasattr(parts[0], 'trim') else parts[0].strip()
                # named imports
                nm = re.search(r"\{([^}]*)\}", clause)
                if nm:
                    raw = [s.strip() for s in (nm.group(1) or '').split(',') if s.strip()]
                    for r in raw:
                        if ' as ' in r:
                            orig, al = r.split(' as ', 1)
                            names.append(orig.strip())
                            alias[orig.strip()] = al.strip()
                        else:
                            names.append(r)
            if names or alias:
                import_symbols.setdefault(str(p), {}).setdefault(key, {'names': [], 'aliases': {}})
                entry = import_symbols[str(p)][key]
                for n in names:
                    if n not in entry['names']:
                        entry['names'].append(n)
                for k2, v2 in alias.items():
                    entry['aliases'][k2] = v2
        if mods:
            imports[str(p)] = mods
        # re-exports
        for m in _RE_EXPORT_NAMED_FROM.finditer(txt):
            names = [s.strip() for s in (m.group(1) or '').split(',') if s.strip()]
            frm = (m.group(2) or '').strip()
            target = _resolve_module(frm, p, root, base_url=base_url, paths=paths)
            reexports.setdefault(str(p), []).append({'from': str(target) if target else frm, 'names': names, 'star': False})
        for m in _RE_EXPORT_ALL_FROM.finditer(txt):
            frm = (m.group(1) or '').strip()
            target = _resolve_module(frm, p, root, base_url=base_url, paths=paths)
            reexports.setdefault(str(p), []).append({'from': str(target) if target else frm, 'names': [], 'star': True})
    return { 'defs': defs, 'imports': imports, 'import_symbols': import_symbols, 'reexports': reexports }


def _out_dir(root: Path) -> Path:
    d = Path(root) / 'out' / 'index'
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def build_and_save(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    idx = build(root)
    blob = { 'version': 1, 'sigs': _sig(root), 'index': idx }
    out = _out_dir(root) / 'js_index.json'
    try:
        out.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
    return idx


def query(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    out = _out_dir(root) / 'js_index.json'
    if not out.exists():
        return build_and_save(root)
    try:
        return (json.loads(out.read_text(encoding='utf-8')) or {}).get('index') or {}
    except Exception:
        return build_and_save(root)


def affected_files_for_symbol(symbol: str, root: Path, *, depth: int = 1) -> List[str]:
    idx = query(root)
    defs: List[str] = list((idx.get('defs') or {}).get(symbol, []) or [])
    if not defs:
        return []
    imports: Dict[str, List[str]] = idx.get('imports') or {}
    import_symbols: Dict[str, Dict[str, Dict[str, Any]]] = idx.get('import_symbols') or {}
    reexports: Dict[str, List[Dict[str, Any]]] = idx.get('reexports') or {}
    # Build rev map: module -> set(importers that import symbol specifically when possible)
    rev: Dict[str, Set[str]] = {}
    for f, mods in imports.items():
        for m in mods or []:
            # prefer symbol-aware check
            sym = (import_symbols.get(f) or {}).get(m)
            if sym:
                names = set(sym.get('names') or [])
                if (symbol in names) or ('default' in names and symbol == 'default'):
                    rev.setdefault(m, set()).add(f)
            else:
                rev.setdefault(m, set()).add(f)
    # Re-exports: if a file re-exports from module, treat it like an importer edge too
    for f, items in (reexports or {}).items():
        for it in items or []:
            m = it.get('from')
            if not m:
                continue
            names = set(it.get('names') or [])
            if (not names) or (symbol in names) or (symbol == 'default' and 'default' in names):
                rev.setdefault(m, set()).add(f)
    out: Set[str] = set(defs)
    frontier: Set[str] = set(defs)
    d = max(0, int(depth))
    for _ in range(d):
        nxt: Set[str] = set()
        for f in frontier:
            for imp in rev.get(f, set()):
                if imp not in out:
                    out.add(imp)
                    nxt.add(imp)
        frontier = nxt
        if not frontier:
            break
    return sorted(out)


def affected_importers_for_file(file_path: str, root: Path, *, depth: int = 1) -> List[str]:
    idx = query(root)
    imports: Dict[str, List[str]] = idx.get('imports') or {}
    rev: Dict[str, Set[str]] = {}
    for f, mods in imports.items():
        for m in mods or []:
            rev.setdefault(m, set()).add(f)
    out: Set[str] = set([file_path])
    frontier: Set[str] = set([file_path])
    d = max(0, int(depth))
    for _ in range(d):
        nxt: Set[str] = set()
        for f in frontier:
            for imp in rev.get(f, set()):
                if imp not in out:
                    out.add(imp)
                    nxt.add(imp)
        frontier = nxt
        if not frontier:
            break
    out.discard(file_path)
    return sorted(out)


def importer_details_for_symbol(symbol: str, root: Path) -> List[Dict[str, Any]]:
    idx = query(root)
    imports: Dict[str, List[str]] = idx.get('imports') or {}
    import_symbols: Dict[str, Dict[str, Dict[str, Any]]] = idx.get('import_symbols') or {}
    details: List[Dict[str, Any]] = []
    for f, mods in imports.items():
        for m in mods or []:
            sym = (import_symbols.get(f) or {}).get(m)
            if not sym:
                continue
            names = sym.get('names') or []
            aliases = sym.get('aliases') or {}
            if (symbol in names) or (symbol == 'default' and 'default' in names):
                local = aliases.get(symbol) if symbol != 'default' else aliases.get('default')
                details.append({'importer': f, 'module': m, 'imported': symbol, 'local': local or symbol})
    return details
