"""Lightweight symbol + import graph utilities.

Builds on selfcoder.analysis.symbols to provide transitive impact estimates for
renames/refactors and a compact index useful for previews.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Set

from . import symbols as _syms


def build(root: Path) -> Dict[str, object]:
    """Return a combined index with defs/uses/reverse and an import graph.
    Keys: defs, uses, reverse, imports
    """
    idx = _syms.build_defs_uses(root, use_cache=True)
    imports = _syms.build_import_graph(root)
    return {**idx, "imports": {str(k): v for k, v in imports.items()}}


def affected_files_for_symbol(symbol: str, root: Path, *, transitive: bool = True, depth: int = 1) -> List[str]:
    """Return a list of files likely affected by a symbol rename.

    Heuristics:
      - Start with files that define or use the symbol.
      - If transitive: include files that import those files' modules (1-hop).
    """
    idx = _syms.build_defs_uses(root, use_cache=True)
    defs = [Path(p) for p in (idx.get("defs", {}).get(symbol) or [])]
    uses = [Path((u or {}).get("file")) for u in (idx.get("uses", {}).get(symbol) or [])]
    base: Set[Path] = set(defs + uses)
    if not transitive or not base:
        return sorted({str(p) for p in base})
    # Multi-hop reverse import expansion
    imap = _syms.build_import_graph(root)
    # Compute reverse map: module path -> files that import it
    rev: Dict[Path, Set[Path]] = {}
    for f, mods in imap.items():
        for m in mods:
            mod_path = root.joinpath(*m.split(".")).with_suffix(".py")
            if mod_path.exists():
                rev.setdefault(mod_path, set()).add(f)
    out: Set[Path] = set(base)
    frontier: Set[Path] = set(base)
    max_depth = max(1, int(depth))
    for _ in range(max_depth):
        next_frontier: Set[Path] = set()
        for p in list(frontier):
            for importer in rev.get(p, set()):
                if importer not in out:
                    out.add(importer)
                    next_frontier.add(importer)
        frontier = next_frontier
        if not frontier:
            break
    return sorted({str(p) for p in out})


__all__ = ["build", "affected_files_for_symbol"]
