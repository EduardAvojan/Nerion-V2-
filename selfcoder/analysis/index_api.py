from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json

from . import symbols as _syms
from . import symbols_graph as _graph


def _out_dir(root: Path) -> Path:
    d = Path(root) / "out" / "index"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _sig_map(root: Path) -> Dict[str, int]:
    return {str(p): int(p.stat().st_mtime_ns) for p in _syms._iter_pyfiles(Path(root))}


def build(root: Path, *, use_cache: bool = True) -> Dict[str, Any]:
    root = Path(root)
    outp = _out_dir(root) / "index.json"
    sigs = _sig_map(root)
    if use_cache and outp.exists():
        try:
            blob = json.loads(outp.read_text(encoding="utf-8"))
            if blob.get("sigs") == sigs and blob.get("version") == 1:
                return blob.get("index", {})
        except Exception:
            pass
    import time as _t
    t0 = _t.time()
    idx = _graph.build(root)
    built_ms = int((_t.time() - t0) * 1000)
    blob = {"version": 1, "sigs": sigs, "index": idx, "stats": {"files": len(sigs), "built_ms": built_ms}}
    try:
        outp.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return idx


def query(root: Path) -> Dict[str, Any]:
    root = Path(root)
    outp = _out_dir(root) / "index.json"
    if not outp.exists():
        return build(root, use_cache=False)
    try:
        blob = json.loads(outp.read_text(encoding="utf-8"))
        return blob.get("index", {})
    except Exception:
        return build(root, use_cache=False)


def affected(root: Path, symbol: str, *, transitive: bool = True) -> List[str]:
    return _graph.affected_files_for_symbol(symbol, Path(root), transitive=transitive)


__all__ = ["build", "query", "affected"]
