
from __future__ import annotations

import argparse
import json


def _import_journal():
    try:
        from app import journal as _journal
        return _journal
    except Exception as e:
        print(f"[journal] unavailable: {e}")
        return None


def cmd_journal_tail(args: argparse.Namespace) -> int:
    _j = _import_journal()
    if not _j or not hasattr(_j, "tail"):
        return 1
    n = int(getattr(args, "n", 20) or 20)
    rows = _j.tail(n=n)
    for r in rows:
        try:
            print(json.dumps(r, ensure_ascii=False))
        except Exception:
            print(str(r))
    return 0


def cmd_journal_day(args: argparse.Namespace) -> int:
    _j = _import_journal()
    if not _j or not hasattr(_j, "by_day"):
        return 1
    day = getattr(args, "date", None)
    if not day:
        print("[journal] day requires a date like 2025-08-16")
        return 2
    rows = _j.by_day(day)
    for r in rows:
        try:
            print(json.dumps(r, ensure_ascii=False))
        except Exception:
            print(str(r))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p_journal = sub.add_parser("journal", help="inspect the system journal")
    p_journal_sub = p_journal.add_subparsers(dest="journal_cmd", required=True)

    pj_tail = p_journal_sub.add_parser("tail", help="show the last N journal entries")
    pj_tail.add_argument("-n", type=int, default=20, help="number of entries to show")
    pj_tail.set_defaults(func=cmd_journal_tail)

    pj_day = p_journal_sub.add_parser("day", help="show entries for a specific day (YYYY-MM-DD)")
    pj_day.add_argument("date", help="date prefix, e.g., 2025-08-16")
    pj_day.set_defaults(func=cmd_journal_day)

