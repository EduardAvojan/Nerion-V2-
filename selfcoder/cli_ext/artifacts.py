from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def _list_artifacts() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    # Site queries
    sq_dir = Path('out/knowledge/site_queries')
    if sq_dir.exists():
        for fp in sorted(sq_dir.glob('*.json'), reverse=True):
            try:
                st = fp.stat()
                items.append({
                    'type': 'site_query',
                    'path': fp.as_posix(),
                    'ts': int(st.st_mtime),
                    'size': st.st_size,
                })
            except Exception:
                continue
    # Chunks
    ch_dir = Path('out/knowledge/chunks')
    if ch_dir.exists():
        for fp in sorted(ch_dir.glob('*.json'), reverse=True):
            try:
                st = fp.stat()
                items.append({
                    'type': 'chunk',
                    'path': fp.as_posix(),
                    'ts': int(st.st_mtime),
                    'size': st.st_size,
                })
            except Exception:
                continue
    return items


def _pretty_show(data: Dict[str, Any]) -> str:
    # Site-query: prefer headline/recommendation/winner and citations
    head = data.get('headline') or data.get('title') or ''
    rec = data.get('recommendation') or ''
    win = (data.get('consensus') or {}).get('winner', {}).get('name') if isinstance(data.get('consensus'), dict) else ''
    lines: List[str] = []
    if head:
        lines.append(f"Headline: {head}")
    if win:
        lines.append(f"Winner: {win}")
    if rec:
        lines.append(f"Recommendation: {rec}")
    # If it has a snippet/text field, show a short preview
    snip = data.get('snippet') or data.get('text') or ''
    if snip:
        lines.append("---")
        lines.append(str(snip)[:400])
    # Citations
    cits = []
    try:
        for c in (data.get('citations') or []):
            if isinstance(c, dict) and (c.get('source') == 'external'):
                u = c.get('url') or ''
                if u:
                    cits.append(u)
    except Exception:
        pass
    if cits:
        lines.append("---")
        lines.append("Citations:")
        for u in cits[:8]:
            lines.append(f" - {u}")
    return "\n".join(lines) if lines else json.dumps(data, ensure_ascii=False, indent=2)


def cmd_list(args: argparse.Namespace) -> int:
    items = _list_artifacts()
    if getattr(args, 'json', False):
        print(json.dumps({'artifacts': items[: int(getattr(args, 'limit', 50) or 50)]}, indent=2))
        return 0
    limit = int(getattr(args, 'limit', 20) or 20)
    for it in items[:limit]:
        print(f"[{it['type']}] {it['path']}  ({it['size']} bytes)")
    if not items:
        print("[artifacts] none found")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    path = getattr(args, 'path', None)
    if not path:
        print("Usage: nerion artifacts show --path <file.json>")
        return 1
    fp = Path(path)
    if not fp.exists():
        print(f"[artifacts] not found: {fp}")
        return 1
    try:
        data = json.loads(fp.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"[artifacts] failed to read: {e}")
        return 1
    text = _pretty_show(data)
    print(text)
    if getattr(args, 'copy_citations', False):
        # best-effort copy external citations to clipboard if available
        try:
            cits = []
            for c in (data.get('citations') or []):
                if isinstance(c, dict) and (c.get('source') == 'external') and c.get('url'):
                    cits.append(c['url'])
            if cits:
                payload = "\n".join(cits)
                if os.name == 'posix' and os.uname().sysname == 'Darwin':
                    import subprocess
                    subprocess.run(['pbcopy'], input=payload.encode('utf-8'), check=False)
                    print("[artifacts] copied citations to clipboard")
                else:
                    # Fallback: just print a block to copy
                    print("---\nCopy these citations:\n" + payload)
        except Exception:
            pass
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('artifacts', help='list and view saved artifacts (site queries, chunks)')
    sp = p.add_subparsers(dest='artifacts_cmd', required=True)

    parser_list = sp.add_parser('list', help='list recent artifacts')
    parser_list.add_argument('--limit', type=int, default=20)
    parser_list.add_argument('--json', action='store_true')
    parser_list.set_defaults(func=cmd_list)

    s = sp.add_parser('show', help='show a specific artifact file')
    s.add_argument('--path', required=True)
    s.add_argument('--copy-citations', action='store_true', help='copy external citation URLs to clipboard (macOS)')
    s.set_defaults(func=cmd_show)

    def _cmd_export(args: argparse.Namespace) -> int:
        from selfcoder.analysis.knowledge.index import search_chunks
        topic = getattr(args, 'topic', None)
        if not topic:
            print('Usage: nerion artifacts export --topic "search:..." [--out report.md]')
            return 1
        chunks = search_chunks(topic=topic, limit=int(getattr(args, 'limit', 20) or 20))
        if not chunks:
            print('[artifacts] no chunks match this topic')
            return 1
        lines = []
        lines.append(f"# Topic: {topic}\n")
        lines.append("## Key Findings")
        for c in chunks[:20]:
            ex = (c.get('extract') or '').strip()
            url = c.get('url') or ''
            dom = ''
            try:
                from urllib.parse import urlparse
                dom = urlparse(url).netloc.replace('www.', '')
            except Exception:
                dom = ''
            head = ex.split('\n', 1)[0][:200] if ex else (dom or url)
            lines.append(f"- {head}")
        lines.append("\n## Citations")
        for c in chunks[:50]:
            u = c.get('url') or ''
            if u:
                lines.append(f"- {u}")
        md = "\n".join(lines) + "\n"
        out = getattr(args, 'out', None)
        if out:
            try:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(md, encoding='utf-8')
                print(f"[artifacts] wrote: {out}")
                return 0
            except Exception as e:
                print(f"[artifacts] write failed: {e}")
                return 1
        print(md)
        return 0

    e = sp.add_parser('export', help='export a topic report (Markdown) from local chunks')
    e.add_argument('--topic', required=True, help='topic name, e.g., "search:laptops"')
    e.add_argument('--out', help='output Markdown file (optional)')
    e.add_argument('--limit', type=int, default=20)
    e.set_defaults(func=_cmd_export)

    # Minimal TUI (curses) to browse artifacts and view details
    def _cmd_tui(args: argparse.Namespace) -> int:
        items = _list_artifacts()
        if not items:
            print('[artifacts] none found')
            return 0
        try:
            import curses
        except Exception:
            print('[artifacts] TUI unavailable (install curses)')
            return 1
        def _run(stdscr):
            curses.curs_set(0)
            idx = 0
            while True:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                stdscr.addstr(0, 0, 'Artifacts (j/k or arrows to move, Enter to show, q to quit)')
                for i, it in enumerate(items[:h-2]):
                    mark = '>' if i == idx else ' '
                    line = f"{mark} [{it['type']}] {it['path']}"
                    stdscr.addnstr(1+i, 0, line, w-1)
                ch = stdscr.getch()
                if ch in (ord('q'), 27):
                    break
                if ch in (curses.KEY_DOWN, ord('j')):
                    idx = min(idx+1, len(items)-1)
                elif ch in (curses.KEY_UP, ord('k')):
                    idx = max(idx-1, 0)
                elif ch in (10, 13):
                    # Show selected
                    fp = Path(items[idx]['path'])
                    try:
                        data = json.loads(fp.read_text(encoding='utf-8'))
                        text = _pretty_show(data)
                    except Exception as e:
                        text = f"[error] {e}"
                    # Simple pager
                    stdscr.erase()
                    lines = text.splitlines() or ['(empty)']
                    top = 0
                    while True:
                        stdscr.erase()
                        stdscr.addnstr(0,0, f"{fp}", w-1)
                        for i in range(1, h-1):
                            j = top + i - 1
                            if j >= len(lines):
                                break
                            stdscr.addnstr(i, 0, lines[j], w-1)
                        stdscr.addnstr(h-1, 0, '[Up/Down to scroll, b to back]', w-1)
                        c2 = stdscr.getch()
                        if c2 in (ord('b'), 27):
                            break
                        if c2 == curses.KEY_DOWN:
                            top = min(top+1, max(0, len(lines)-h))
                        if c2 == curses.KEY_UP:
                            top = max(0, top-1)
        curses.wrapper(_run)
        return 0

    t = sp.add_parser('tui', help='interactive terminal browser for artifacts')
    t.set_defaults(func=_cmd_tui)
