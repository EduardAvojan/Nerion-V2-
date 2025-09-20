from __future__ import annotations

import argparse
import json
from typing import Any

from app.chat.memory_bridge import LongTermMemory


def _load_memory(path: str) -> LongTermMemory:
    return LongTermMemory(path=path)


def _cmd_show(args: argparse.Namespace) -> int:
    mem = _load_memory(args.path)
    items = mem.list_memories()
    print(json.dumps(items[: args.k], indent=2, ensure_ascii=False))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    mem = _load_memory(args.path)
    results = mem.find_relevant(args.query, k=args.k)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def register(subparsers: Any) -> None:
    p = subparsers.add_parser("memory", help="inspect or search long-term memory")
    sp = p.add_subparsers(dest="memory_cmd", required=True)

    p_show = sp.add_parser("show", help="show stored memories (top-k)")
    p_show.add_argument("--path", default="memory_db.json", help="path to memory DB (default: memory_db.json)")
    p_show.add_argument("--k", type=int, default=20, help="number of items to display")
    p_show.set_defaults(func=_cmd_show)

    p_search = sp.add_parser("search", help="search memories for a query")
    p_search.add_argument("query", help="search query string")
    p_search.add_argument("--path", default="memory_db.json", help="path to memory DB (default: memory_db.json)")
    p_search.add_argument("--k", type=int, default=10, help="number of matches to return")
    p_search.set_defaults(func=_cmd_search)
