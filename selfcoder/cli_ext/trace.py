from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def _load_last(path: Path, n: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for ln in reversed(lines):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
        if len(out) >= n:
            break
    return list(reversed(out))


def cmd_last(args: argparse.Namespace) -> int:
    n = int(getattr(args, 'last', 20) or 20)
    rows = _load_last(Path('out/experience/log.jsonl'), n)
    if not rows:
        print('[trace] no records')
        return 0
    # Summarize by tool (from action_taken.steps)
    from collections import defaultdict
    tools = defaultdict(list)
    for r in rows:
        for st in ((r.get('action_taken') or {}).get('steps') or []):
            t = st.get('tool') or 'unknown'
            d = int(st.get('duration_ms') or 0)
            tools[t].append(d)
    top = sorted(((k, sum(v), len(v)) for k, v in tools.items()), key=lambda x: x[1], reverse=True)[:10]
    # Count per-tool successes (from experience log)
    ok_by_tool = {}
    for r in rows:
        if not r.get('outcome_success'):
            continue
        for st in ((r.get('action_taken') or {}).get('steps') or []):
            t = st.get('tool') or 'unknown'
            ok_by_tool[t] = ok_by_tool.get(t, 0) + 1
    top_ok = sorted(((k, v) for k, v in ok_by_tool.items()), key=lambda kv: kv[1], reverse=True)[:5]

    # Build report
    rep = {
        'records': len(rows),
        'top_tools': [
            {'tool': k, 'calls': c, 'total_ms': s, 'avg_ms': int(s/max(1,c))}
            for k, s, c in top
        ],
        'top_ok_tools': [{'tool': k, 'ok': c} for k, c in top_ok]
    }
    if getattr(args, 'json', False):
        print(json.dumps(rep, indent=2))
        return 0
    # Human summary by mode
    mode = str(getattr(args, 'mode', 'time') or 'time').lower()
    if mode == 'ok':
        print('[trace] top tools by OK count')
        print(' Tool'.ljust(28) + 'OK')
        print(' ' + '-'*26 + ' ' + '-'*6)
        for t in rep['top_ok_tools']:
            nm = str(t.get('tool') or '')[:24]
            ok = str(int(t.get('ok') or 0))
            print(f" {nm.ljust(26)} {ok.rjust(6)}")
    else:
        print('[trace] top tools by total time')
        print(' Tool'.ljust(24) + 'Calls'.rjust(8) + ' Total(ms)'.rjust(12) + ' Avg(ms)'.rjust(10))
        print(' ' + '-'*22 + ' ' + '-'*6 + ' ' + '-'*10 + ' ' + '-'*8)
        for t in rep['top_tools']:
            nm = str(t.get('tool') or '')[:20]
            calls = str(int(t.get('calls') or 0))
            tot = str(int(t.get('total_ms') or 0))
            avg = str(int(t.get('avg_ms') or 0))
            print(f" {nm.ljust(20)} {calls.rjust(6)} {tot.rjust(10)} {avg.rjust(8)}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('trace', help='inspect recent runtime traces (experience log)')
    sp = p.add_subparsers(dest='trace_cmd', required=True)
    last = sp.add_parser('last', help='summarize last N records')
    last.add_argument('--last', type=int, default=20)
    last.add_argument('--mode', choices=['time','ok'], default='time', help='summary mode (time or ok)')
    last.add_argument('--json', action='store_true', help='emit JSON instead of human table')
    last.set_defaults(func=cmd_last)

    dg = sp.add_parser('digest', help='daily digest style summary over a time window (default 24h)')
    dg.add_argument('--since', default='24h', help='time window like 24h/60m/1h (default 24h)')
    dg.add_argument('--json', action='store_true')
    dg.set_defaults(func=cmd_digest)

    ex = sp.add_parser('export', help='export a markdown digest of recent actions')
    ex.add_argument('--since', default='24h')
    ex.add_argument('--out', required=True)
    ex.set_defaults(func=cmd_export)

def summarize_since(lines, since_sec: int = 24*3600):
    """Summarize experience log JSON lines within a time window.

    Returns a dict with keys: actions, blocked, time_ms.
    A line is counted when its ts >= now - since_sec.
    """
    import time as _time
    import json as _json
    cutoff = _time.time() - since_sec
    summary = {"actions": 0, "blocked": 0, "time_ms": 0}
    for row in lines or []:
        try:
            obj = _json.loads(row)
        except Exception:
            continue
        ts = obj.get("ts", 0)
        if ts < cutoff:
            continue
        summary["actions"] += 1
        if obj.get("result") in ("BLOCKED", "FAIL", "ERROR"):
            summary["blocked"] += 1
        summary["time_ms"] += int(obj.get("ms", 0) or 0)
    return summary


def _parse_since(s: str) -> int:
    s = (s or '').strip().lower()
    import re as _re
    m = _re.fullmatch(r"(\d+)(s|m|h)?", s)
    if not m:
        return 24*3600
    n = int(m.group(1))
    unit = m.group(2) or 's'
    if unit == 'h':
        return n * 3600
    if unit == 'm':
        return n * 60
    return n


def cmd_digest(args: argparse.Namespace) -> int:
    since_s = _parse_since(getattr(args, 'since', '24h'))
    path = Path('out/experience/log.jsonl')
    if not path.exists():
        print('[trace] no records')
        return 0
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()[-2000:]
    summary = summarize_since(lines, since_sec=since_s)
    if getattr(args, 'json', False):
        import json as _json
        print(_json.dumps(summary, indent=2))
    else:
        print(f"[digest] actions={summary['actions']} blocked={summary['blocked']} time_ms={summary['time_ms']}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    since_s = _parse_since(getattr(args, 'since', '24h'))
    path = Path('out/experience/log.jsonl')
    lines = []
    if path.exists():
        try:
            lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()[-2000:]
        except Exception:
            lines = []
    summary = summarize_since(lines, since_sec=since_s)
    out_path = Path(getattr(args, 'out'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        '# Nerion Trace Digest',
        '',
        f"Since: {getattr(args, 'since', '24h')}",
        '',
        f"- Actions: {summary['actions']}",
        f"- Blocked: {summary['blocked']}",
        f"- Time (ms): {summary['time_ms']}",
        '',
        'Generated by `nerion trace export`',
    ]
    out_path.write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({'ok': True, 'out': str(out_path)}, indent=2))
    return 0
