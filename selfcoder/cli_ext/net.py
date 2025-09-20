from __future__ import annotations

import argparse
import json

from ops.security.net_gate import NetworkGate
from core.ui.messages import fmt as _fmt_msg
from core.ui.messages import Result as _MsgRes
import time as _t


def _status_dict() -> dict:
    try:
        st = NetworkGate.state().name
    except Exception:
        st = "UNKNOWN"
    try:
        tl = NetworkGate.time_remaining()
    except Exception:
        tl = None
    return {
        "state": st,
        "time_remaining_s": tl,
    }


def cmd_status(_args: argparse.Namespace) -> int:
    data = _status_dict()
    print(json.dumps(data, indent=2))
    st = str(data.get('state'))
    print(_fmt_msg('net', 'status', _MsgRes.OK if st == 'SESSION' else _MsgRes.SKIP, f"state={st}"))
    return 0


def cmd_prefs(args: argparse.Namespace) -> int:
    # Bridge to app/chat/net_access helpers for persistence
    from app.chat.net_access import load_net_prefs, save_net_prefs
    if getattr(args, "clear", False):
        prefs = {"always_allow_by_task": {}}
        save_net_prefs(prefs)
        print(_fmt_msg('net', 'prefs', _MsgRes.OK, 'cleared'))
        return 0
    prefs = load_net_prefs()
    print(json.dumps(prefs, indent=2))
    print(_fmt_msg('net', 'prefs', _MsgRes.OK, 'loaded'))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("net", help="network policy helpers")
    sp = p.add_subparsers(dest="net_cmd", required=True)

    s = sp.add_parser("status", help="show current session state and time remaining")
    s.set_defaults(func=cmd_status)

    pr = sp.add_parser("prefs", help="inspect or clear persisted network preferences")
    pr.add_argument("--clear", action="store_true", help="clear saved 'always allow' preferences")
    pr.set_defaults(func=cmd_prefs)

    # Offline-only window for the current session (suppresses prompts)
    def _parse_dur(spec: str) -> int:
        if not spec:
            return 0
        s = spec.strip().lower()
        import re
        m = re.match(r"^(\d+)(s|m|h)$", s)
        if not m:
            return 0
        n = int(m.group(1))
        u = m.group(2)
        if u == 's':
            return n
        if u == 'm':
            return n * 60
        if u == 'h':
            return n * 3600
        return 0

    def _cmd_offline(args: argparse.Namespace) -> int:
        from app.chat.net_access import load_net_prefs, save_net_prefs
        secs = _parse_dur(getattr(args, 'for_', None) or '')
        if secs <= 0:
            print("Usage: nerion net offline --for 30m (supports s/m/h)")
            return 1
        prefs = load_net_prefs()
        until = int(_t.time()) + secs
        prefs.setdefault('offline_until', until)
        prefs['offline_until'] = until
        save_net_prefs(prefs)
        print(json.dumps({"offline_until": until}, indent=2))
        print(_fmt_msg('net', 'offline', _MsgRes.OK, f"for={getattr(args,'for_', '')}"))
        return 0

    off = sp.add_parser("offline", help="force offline mode for a duration (suppresses prompts)")
    off.add_argument("--for", dest="for_", required=True, help="duration like 30m, 2h, 45s")
    off.set_defaults(func=_cmd_offline)
