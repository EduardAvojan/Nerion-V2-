from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfcoder.analysis import symbols as syms
from selfcoder.analysis import symbols_graph as sgraph
from selfcoder.analysis import index_api as idxapi
from selfcoder.analysis import js_index as jsidx


def cmd_affected(args: argparse.Namespace) -> int:
    root = Path(getattr(args, "root", ".")).resolve()
    # Ensure index on disk
    idxapi.build(root, use_cache=True)
    idx = syms.build_defs_uses(root, use_cache=True)
    symbol = getattr(args, "symbol", None)
    out = {"defs": {}, "uses": {}, "affected": []}
    if symbol:
        defs = [p for p in (idx.get("defs", {}).get(symbol) or [])]
        uses = [u for u in (idx.get("uses", {}).get(symbol) or [])]
        aff = sgraph.affected_files_for_symbol(symbol, root, transitive=bool(getattr(args, "transitive", False)))
        out = {"defs": {symbol: defs}, "uses": {symbol: uses}, "affected": aff}
    else:
        out = {"defs": idx.get("defs", {}), "uses": idx.get("uses", {})}
    print(json.dumps(out, indent=2))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("graph", help="Code graph utilities")
    sp = p.add_subparsers(dest="graph_cmd", required=True)

    aff = sp.add_parser("affected", help="Show defs/uses and affected files for a symbol (or all if omitted)")
    aff.add_argument("--symbol", help="symbol name (function/class)")
    aff.add_argument("--root", default=".")
    aff.add_argument("--transitive", action="store_true", help="Include reverse importers (set depth>1 for multi-hop)")
    aff.add_argument("--depth", type=int, default=1, help="Reverse importer depth (default 1)")
    aff.add_argument("--json", action="store_true", help="Output JSON only")
    aff.add_argument("--methods", action="store_true", help="Include class methods as Class.method symbols")
    aff.add_argument("--js", action="store_true", help="Use JS/TS index instead of Python index")
    aff.add_argument("--details", action="store_true", help="When --js, include importer details with alias/local names")
    def _run_aff(args: argparse.Namespace) -> int:
        root = Path(getattr(args, "root", ".")).resolve()
        symbol = getattr(args, "symbol", None)
        if getattr(args, 'js', False):
            # JS/TS path
            idx = jsidx.build_and_save(root)
            depth = int(getattr(args, 'depth', 1) or 1)
            aff = jsidx.affected_files_for_symbol(symbol or '', root, depth=depth) if symbol else []
            risk_radius = max(0, len(aff) - len((idx.get('defs') or {}).get(symbol or '', []))) if symbol else 0
            out = {"defs": (idx.get('defs') or {}).get(symbol or '', []), "affected": aff, "risk_radius": risk_radius}
            if symbol and getattr(args, 'details', False):
                out['importers'] = jsidx.importer_details_for_symbol(symbol, root)
        else:
            # Python path
            idxapi.build(root, use_cache=True)
            idx = syms.build_defs_uses(root, use_cache=True, include_methods=bool(getattr(args, 'methods', False)))
            if symbol:
                defs = [p for p in (idx.get("defs", {}).get(symbol) or [])]
                uses = [u for u in (idx.get("uses", {}).get(symbol) or [])]
                depth = int(getattr(args, 'depth', 1) or 1)
                aff = sgraph.affected_files_for_symbol(symbol, root, transitive=bool(getattr(args, "transitive", False) or depth > 0), depth=depth)
                out = {"defs": {symbol: defs}, "uses": {symbol: uses}, "affected": aff}
            else:
                out = {"defs": idx.get("defs", {}), "uses": idx.get("uses", {})}
        if getattr(args, 'json', False):
            print(json.dumps(out, indent=2))
        else:
            # Pretty print
            if getattr(args, 'js', False):
                sym = symbol or '<symbol>'
                print(f"[graph.js] Defs/Affected for: {sym}")
                d = out.get('defs', [])
                print("Defs:")
                for p in d or []:
                    print(f"  - {p}")
                depth = int(getattr(args, 'depth', 1) or 1)
                print(f"Affected (depth {depth}):")
                for p in out.get('affected', []) or []:
                    print(f"  - {p}")
                print(f"Risk radius: {out.get('risk_radius', 0)}")
                if getattr(args, 'details', False):
                    print('Importers:')
                    for it in out.get('importers', []) or []:
                        print(f"  - {it.get('importer')} imports {it.get('imported')} as {it.get('local')} from {it.get('module')}")
            else:
                sym = symbol or '<all>'
                print(f"[graph] Defs/Uses for: {sym}")
                if symbol:
                    d = out.get('defs', {}).get(symbol, [])
                    u = out.get('uses', [])
                else:
                    d = out.get('defs', {})
                    u = out.get('uses', {})
                if symbol:
                    print("Defs:")
                    for p in d or []:
                        print(f"  - {p}")
                    print("Uses:")
                    for rec in u or []:
                        print(f"  - {rec.get('file')}:{rec.get('line')}")
                    depth = int(getattr(args, 'depth', 1) or 1)
                    print(f"Affected (depth {depth}):")
                    for p in out.get('affected', []) or []:
                        print(f"  - {p}")
                else:
                    print(f"  symbols: {len(d)}; uses-groups: {len(u)}")
        return 0
    aff.set_defaults(func=_run_aff)
