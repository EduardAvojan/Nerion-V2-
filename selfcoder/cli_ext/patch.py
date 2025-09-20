from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import difflib
import hashlib
import os

from ops.security import fs_guard

# --- Router/profile helpers (lazy imports inside functions) ---------------
def _route_for_planfile(planfile: Path) -> None:
    try:
        data = json.loads(planfile.read_text(encoding='utf-8'))
    except Exception:
        return
    try:
        target = None
        if isinstance(data, dict):
            target = data.get('target_file') or (data.get('files')[0] if (data.get('files') or []) else None)
        # Apply task-aware router and a conservative profile scope
        try:
            from selfcoder.llm_router import apply_router_env as _route
            _route(instruction=None, file=str(target) if target else None, task='code')
        except Exception:
            pass
        try:
            from selfcoder.policy.profile_resolver import decide as _dec, apply_env_scoped as _scope
            # Use 'apply_plan' signal since patch operates on existing plans
            dec = _dec('apply_plan', signals={'files_count': 1 if target else 0, 'kinds_ast_only': True})
            # Best-effort scope (will only set unset env)
            _scope(dec).__enter__()  # leak ok; patch commands are short-lived
        except Exception:
            pass
    except Exception:
        pass


def cmd_preview(args: argparse.Namespace) -> int:
    from selfcoder.orchestrator import _apply_actions_preview as preview, _unified_diff_for_file as udiff
    try:
        from core.ui.diff import render_unified as _render_simple
    except Exception:
        _render_simple = None
    planfile = Path(args.planfile)
    _route_for_planfile(planfile)
    plan = json.loads(planfile.read_text(encoding='utf-8'))
    actions = plan.get('actions') or []
    files: List[Path] = []
    if plan.get('files'):
        files = [Path(p) for p in plan['files']]
    elif plan.get('target_file'):
        files = [Path(plan['target_file'])]
    pv = preview(files, actions)
    if getattr(args, 'json', False):
        out = {str(p): udiff(p, old, new) for p, (old, new) in pv.items()}
        print(json.dumps({'diffs': out}, ensure_ascii=False, indent=2))
        return 0
    use_simple = bool(getattr(args, 'simple_diff', False)) and callable(_render_simple)
    for p, (old, new) in pv.items():
        if use_simple:
            a, b, ops = _diff_opcodes(old, new)
            lines = []
            for tag, i1, i2, j1, j2 in ops:
                if tag == 'equal':
                    for ln in a[i1:i2]:
                        lines.append((' ', ln.rstrip('\n')))
                elif tag == 'replace' or tag == 'delete':
                    for ln in a[i1:i2]:
                        lines.append(('-', ln.rstrip('\n')))
                if tag == 'replace' or tag == 'insert':
                    for ln in b[j1:j2]:
                        lines.append(('+', ln.rstrip('\n')))
            print(f"=== {p} ===")
            print(_render_simple(lines))
        else:
            print(udiff(p, old, new))
    return 0


# ---- Small helpers (testable) --------------------------------------------
def _summarize_tool_counts(findings: List[Dict[str, object]]) -> str:
    counts: Dict[str, int] = {}
    for f in (findings or [])[:200]:
        try:
            t = str((f or {}).get('tool') or '').lower()
        except Exception:
            t = ''
        if not t:
            continue
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return ''
    parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    return "Tools: " + " ".join(parts)


def cmd_apply_selected(args: argparse.Namespace) -> int:
    from selfcoder.orchestrator import run_actions_on_files as apply_files
    planfile = Path(args.planfile)
    _route_for_planfile(planfile)
    plan = json.loads(planfile.read_text(encoding='utf-8'))
    actions = plan.get('actions') or []
    files = [Path(f) for f in (getattr(args, 'file', []) or [])]
    if not files:
        print('[patch] specify --file for each file to apply')
        return 1
    changed = apply_files(files, actions, dry_run=False)
    print(json.dumps({'applied': [str(p) for p in changed]}, indent=2))
    return 0 if changed else 1


# ---------------- Hunk-level helpers -----------------

def _diff_opcodes(old: str, new: str):
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, a, b)
    return a, b, sm.get_opcodes()


def _format_hunk_preview(a: List[str], b: List[str], tag: str, i1: int, i2: int, j1: int, j2: int, idx: int) -> str:
    head = f"HUNK {idx} [{tag}] old:{i1}-{i2} -> new:{j1}-{j2}"
    # Show up to first/last 3 changed lines
    old_snip = ''.join(a[i1:i2])[:240]
    new_snip = ''.join(b[j1:j2])[:240]
    body = []
    body.append("--- OLD ---\n" + old_snip.rstrip())
    body.append("+++ NEW ---\n" + new_snip.rstrip())
    return head + "\n" + "\n".join(body)


def cmd_preview_hunks(args: argparse.Namespace) -> int:
    from selfcoder.orchestrator import _apply_actions_preview as preview
    planfile = Path(args.planfile)
    _route_for_planfile(planfile)
    plan = json.loads(planfile.read_text(encoding='utf-8'))
    actions = plan.get('actions') or []
    # Resolve which file to preview
    files: List[Path] = []
    if getattr(args, 'file', None):
        files = [Path(args.file)]
    elif plan.get('files'):
        files = [Path(p) for p in plan['files']]
    elif plan.get('target_file'):
        files = [Path(plan['target_file'])]
    if not files:
        print('[patch] preview-hunks: specify --file or include target in plan')
        return 1
    pv = preview(files, actions)
    if not pv:
        print('[patch] no predicted changes for selected file(s)')
        return 1
    for p, (old, new) in pv.items():
        print(f"=== {p} ===")
        a, b, ops = _diff_opcodes(old, new)
        idx = 0
        for tag, i1, i2, j1, j2 in ops:
            if tag == 'equal':
                continue
            print(_format_hunk_preview(a, b, tag, i1, i2, j1, j2, idx))
            print()
            idx += 1
    return 0


def _apply_selected_hunks(old: str, new: str, selected: List[int]) -> str:
    a, b, ops = _diff_opcodes(old, new)
    out: List[str] = []
    idx = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal':
            out.extend(a[i1:i2])
            continue
        if idx in selected:
            out.extend(b[j1:j2])
        else:
            out.extend(a[i1:i2])
        idx += 1
    text = ''.join(out)
    if not text.endswith('\n'):
        text += '\n'
    return text


def cmd_apply_hunks(args: argparse.Namespace) -> int:
    from selfcoder.orchestrator import _apply_actions_preview as preview
    planfile = Path(args.planfile)
    _route_for_planfile(planfile)
    plan = json.loads(planfile.read_text(encoding='utf-8'))
    actions = plan.get('actions') or []
    if not getattr(args, 'file', None):
        print('[patch] apply-hunks: --file is required')
        return 1
    target = Path(args.file)
    pv = preview([target], actions)
    pair = pv.get(target)
    if pair is None:
        print('[patch] no predicted changes for this file')
        return 1
    old, new = pair
    selected = sorted(set(int(h) for h in (getattr(args, 'hunk', []) or [])))
    if not selected:
        print('[patch] specify at least one --hunk index (from preview-hunks output)')
        return 1
    patched = _apply_selected_hunks(old, new, selected)
    # Repo-safe write (infer project root from target path automatically)
    safe_path = fs_guard.ensure_in_repo_auto(str(target))
    safe_path.write_text(patched, encoding='utf-8')
    print(json.dumps({'applied_file': safe_path.as_posix(), 'hunks': selected}, indent=2))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('patch', help='preview or apply selected parts of a plan (surgical)')
    sp = p.add_subparsers(dest='patch_cmd', required=True)

    pr = sp.add_parser('preview', help='print unified diff for a plan without applying')
    pr.add_argument('planfile')
    pr.add_argument('--json', action='store_true')
    pr.add_argument('--simple-diff', action='store_true', help='use minimal colored diff renderer (no headers)')
    pr.set_defaults(func=cmd_preview)

    ap = sp.add_parser('apply-selected', help='apply a plan only to selected files')
    ap.add_argument('planfile')
    ap.add_argument('--file', action='append', default=[], help='file to apply (may repeat)')
    ap.set_defaults(func=cmd_apply_selected)

    ph = sp.add_parser('preview-hunks', help='preview change hunks per file with indexes')
    ph.add_argument('planfile')
    ph.add_argument('--file', help='single file to preview (defaults to plan target)')
    ph.set_defaults(func=cmd_preview_hunks)

    ah = sp.add_parser('apply-hunks', help='apply only selected hunks for a single file')
    ah.add_argument('planfile')
    ah.add_argument('--file', required=True, help='file to apply hunks to')
    ah.add_argument('--hunk', action='append', default=[], help='hunk index to apply (repeat)')
    ah.set_defaults(func=cmd_apply_hunks)

    # Minimal terminal TUI (curses-based) to select files/hunks and apply
    tui = sp.add_parser('tui', help='interactive TUI to preview/apply selected hunks')
    tui.add_argument('planfile')
    tui.set_defaults(func=cmd_tui)

    # Safe-apply flow: reviewer gate + subset tests, then apply
    sa = sp.add_parser('safe-apply', help='apply a plan only if reviewer and test subset are clean')
    sa.add_argument('planfile')
    sa.add_argument('--file', action='append', default=[], help='limit to specific files (repeatable)')
    sa.add_argument('--json', action='store_true', help='emit a small JSON summary')
    sa.set_defaults(func=cmd_safe_apply)


# ---------------- TUI -----------------

def _build_previews_for_plan(plan_path: Path):
    from selfcoder.orchestrator import _apply_actions_preview as preview
    _route_for_planfile(plan_path)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    actions = plan.get('actions') or []
    files: List[Path] = []
    if plan.get('files'):
        files = [Path(p) for p in plan['files']]
    elif plan.get('target_file'):
        files = [Path(plan['target_file'])]
    pv = preview(files, actions)
    return plan, actions, pv


def cmd_tui(args: argparse.Namespace) -> int:
    try:
        import curses  # type: ignore
    except Exception:
        print('[patch.tui] curses not available in this environment')
        return 1

    planfile = Path(args.planfile)
    plan, actions, pv = _build_previews_for_plan(planfile)
    if not pv:
        print('[patch.tui] no predicted changes to preview')
        return 1

    files = list(pv.keys())
    # Precompute hunks per file
    per_file_hunks: Dict[Path, List[tuple]] = {}
    pairs: Dict[Path, tuple[str, str]] = {}
    for p, (old, new) in pv.items():
        a, b, ops = _diff_opcodes(old, new)
        # store ops and buffers
        per_file_hunks[p] = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in ops if tag != 'equal']
        pairs[p] = (old, new)

    selected: Dict[Path, set[int]] = {p: set() for p in files}
    sel_file_idx = 0
    sel_hunk_idx = 0
    # Load UI prefs
    try:
        from core.ui.prefs import load_prefs, save_prefs
        _prefs = load_prefs() or {}
        _ov = dict((_prefs.get('overlay') or {}))
    except Exception:
        load_prefs = None
        save_prefs = None
        _ov = {}

    show_security = bool(_ov.get('show_security', False))
    security_report: List[str] = []
    # Security overlay toggles/state
    show_js_only = bool(_ov.get('js_only', False))
    show_eslint_only = bool(_ov.get('eslint_only', False))
    show_tsc_only = bool(_ov.get('tsc_only', False))
    last_sec_findings: List[Dict[str, Any]] = []
    last_sec_header: List[str] = []
    sec_file_list: List[str] = []
    sec_file_idx: int = 0
    sec_file_counts: Dict[str, Dict[str, int]] = {}
    # Test subset overlay state
    show_tests = False
    tests_report: List[str] = []
    last_test_sig: Optional[str] = None
    last_test_ok: Optional[bool] = None
    # JS affected importers overlay
    show_js_aff = False
    js_aff_lines: List[str] = []
    # Python rename affected overlay
    show_py_aff = False
    py_aff_lines: List[str] = []

    def _predicted_from_selection() -> Dict[str, str]:
        predicted: Dict[str, str] = {}
        for p, (old, new) in pairs.items():
            if selected[p]:
                patched = _apply_selected_hunks(old, new, sorted(selected[p]))
                predicted[p.as_posix()] = patched
            else:
                predicted[p.as_posix()] = new
        return predicted

    def _sig_predicted(pred: Dict[str, str]) -> str:
        h = hashlib.sha1()
        for k in sorted(pred.keys()):
            v = pred[k] or ""
            h.update(k.encode('utf-8', errors='ignore'))
            h.update(b"\0")
            h.update(v.encode('utf-8', errors='ignore'))
            h.update(b"\n")
        return h.hexdigest()

    def _assess_security(predicted: Dict[str, str]) -> Tuple[bool, List[str]]:
        lines: List[str] = []
        try:
            from selfcoder.reviewers.reviewer import review_predicted_changes
            rep = review_predicted_changes(predicted, Path('.').resolve())
            sec = rep.get('security', {}) if isinstance(rep, dict) else {}
            proceed = bool(sec.get('proceed'))
            score = int(sec.get('score') or 0)
            header = [f"Security: {'OK' if proceed else 'BLOCK'} (score={score})"]
            # Summarize tool counts (eslint/tsc/etc.)
            findings = list(sec.get('findings') or [])
            tc = _summarize_tool_counts(findings)
            if tc:
                header.append(tc)
            # Save full set and header for toggle rendering
            nonlocal last_sec_findings, last_sec_header
            last_sec_findings = findings
            last_sec_header = header
            # Build per-file lists and tool counts
            files_set: List[str] = []
            counts: Dict[str, Dict[str, int]] = {}
            for f in findings:
                fn = str(f.get('filename') or '')
                if not fn:
                    continue
                if fn not in files_set:
                    files_set.append(fn)
                tool = str(f.get('tool') or '').lower()
                d = counts.setdefault(fn, {'eslint':0,'tsc':0,'other':0})
                if tool in {'eslint','tsc'}:
                    d[tool] += 1
                else:
                    d['other'] += 1
            nonlocal sec_file_list, sec_file_counts, sec_file_idx
            sec_file_list = files_set
            sec_file_counts = counts
            if sec_file_idx >= len(sec_file_list):
                sec_file_idx = 0
            # Render default view
            lines.extend(header)
            def _fmt(f):
                return f" - [{f.get('severity')}] {f.get('rule_id')} {f.get('filename')}:{f.get('line')} — {f.get('message')}"
            for f in findings[:6]:
                lines.append(_fmt(f))
            return proceed, lines
        except Exception:
            return True, ["[security] review unavailable"]

    def _write_predicted_into_shadow(shadow_root: Path, predicted: Dict[str, str], repo_root: Path) -> None:
        for fname, text in (predicted or {}).items():
            try:
                p = Path(fname)
                rel = p
                if p.is_absolute():
                    try:
                        rel = p.relative_to(repo_root)
                    except Exception:
                        rel = Path(p.name)
                dest = shadow_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text or '', encoding='utf-8')
            except Exception:
                continue

    def _run_pytest_subset_in_shadow(predicted: Dict[str, str]) -> Tuple[bool, List[str], List[str]]:
        """Return (ok, failing_nodes, brief_lines). Uses smoke tests if present, else full tests."""
        from selfcoder.simulation import make_shadow_copy
        from selfcoder.analysis.failtriage import _extract_failed_tests as _extract
        try:
            from ops.security.safe_subprocess import safe_run
        except Exception:
            import subprocess
            def safe_run(argv, **kwargs):
                return subprocess.run(argv, **{k: v for k, v in kwargs.items() if k in ("cwd","timeout","check","capture_output","text")})

        root = Path('.').resolve()
        shadow = make_shadow_copy(root)
        try:
            _write_predicted_into_shadow(shadow, predicted, root)
            subset_env = (os.getenv('NERION_PATCH_TUI_NODEIDS') or '').strip()
            import sys as _sys
            if subset_env:
                nodeids = [s.strip() for s in subset_env.split(',') if s.strip()]
                args = [_sys.executable, '-m', 'pytest', '-q', '-x', *nodeids]
            else:
                smoke = shadow / 'tests' / 'smoke'
                if smoke.exists():
                    args = [_sys.executable, '-m', 'pytest', '-q', '-x', 'tests/smoke']
                else:
                    args = [_sys.executable, '-m', 'pytest', '-q', '-x']
            res = safe_run(args, cwd=shadow, timeout=int(os.getenv('NERION_PATCH_TUI_PYTEST_TIMEOUT', '300')), check=False, capture_output=True, text=True)
            ok = (res.returncode == 0)
            out = (res.stdout or '') + '\n' + (res.stderr or '')
            failing = _extract(out) if not ok else []
            lines: List[str] = []
            if ok:
                lines.append('[tests] subset passed')
            else:
                lines.append('[tests] subset FAILED')
                for n in failing[:6]:
                    lines.append(f' - {n}')
            return ok, failing, lines
        finally:
            try:
                import shutil
                shutil.rmtree(shadow, ignore_errors=True)
            except Exception:
                pass

    def _compute_security():
        nonlocal security_report
        predicted = _predicted_from_selection()
        proceed, lines = _assess_security(predicted)
        security_report = lines[:50]

    def _compute_js_affected():
        nonlocal js_aff_lines
        try:
            from selfcoder.analysis import js_index as jsidx
            root = Path('.').resolve()
            jsidx.build_and_save(root)
            # Collect predicted JS/TS files changed
            pred = _predicted_from_selection()
            changed = [Path(p) for p,src in pred.items() if str(p).endswith(('.js','.jsx','.ts','.tsx','.mjs','.cjs'))]
            lines: List[str] = []
            for p in changed[:5]:
                aff = jsidx.affected_importers_for_file(str(Path(p).resolve()), root, depth=1)
                if not aff:
                    continue
                lines.append(f"{p}: {len(aff)} importer(s)")
                for imp in aff[:5]:
                    lines.append(f"  - {imp}")
            if not lines:
                lines = ["(no JS/TS importers for selected files)"]
            js_aff_lines = lines[:50]
        except Exception:
            js_aff_lines = ["(JS index unavailable)"]

    def _compute_py_affected():
        nonlocal show_py_aff, py_aff_lines
        try:
            # Detect rename action from plan actions
            rename_from = None
            try:
                for act in (actions or []):
                    if isinstance(act, dict) and (act.get('kind') == 'rename_symbol'):
                        payload = act.get('payload') or {}
                        rename_from = payload.get('from') or payload.get('old')
                        if rename_from:
                            break
            except Exception:
                rename_from = None
            if not rename_from:
                py_aff_lines = ["(no rename action detected)"]
                show_py_aff = True
                return
            from selfcoder.analysis import symbols_graph as sgraph
            root = Path('.').resolve()
            aff = sgraph.affected_files_for_symbol(str(rename_from), root, transitive=True, depth=2)
            if not aff:
                py_aff_lines = ["(no affected files)"]
            else:
                py_aff_lines = [f"{len(aff)} affected files (depth 2):"] + [f"  - {p}" for p in aff[:8]]
            show_py_aff = True
        except Exception:
            py_aff_lines = ["(Python graph unavailable)"]
            show_py_aff = True

    def _apply_selection():
        # Apply only selected hunks per file; if none selected for a file, apply full change
        for p, (old, new) in pairs.items():
            use = new if not selected[p] else _apply_selected_hunks(old, new, sorted(selected[p]))
            safe_p = fs_guard.ensure_in_repo_auto(str(p))
            safe_p.write_text(use, encoding='utf-8')

    def _draw(stdscr):
        nonlocal show_security, security_report
        nonlocal show_js_only, show_eslint_only, show_tsc_only
        nonlocal sec_file_idx
        nonlocal show_tests, tests_report, last_test_sig, last_test_ok
        nonlocal show_js_aff, show_py_aff
        curses.curs_set(0)
        stdscr.nodelay(False)
        height, width = stdscr.getmaxyx()
        left_w = max(24, width // 3)
        while True:
            stdscr.erase()
            # Header
            stdscr.addstr(0, 0, 'Patch TUI — j/k: files  h/l: hunks  space: toggle  t: test  g: gate  a: apply  A: force  q: quit')
            # Files list
            stdscr.addstr(2, 0, 'Files:')
            for i, p in enumerate(files[: height - 6]):
                mark = '>' if i == sel_file_idx else ' '
                sel_count = len(selected[p])
                stdscr.addstr(3 + i, 0, f"{mark} {p.as_posix()} ({sel_count} hunks)")
            # Hunks for selected file
            cur = files[sel_file_idx]
            hunks = per_file_hunks.get(cur) or []
            stdscr.addstr(2, left_w, f'Hunks for {cur.name}:')
            for j, h in enumerate(hunks[: height - 6]):
                tag, i1, i2, j1, j2 = h
                mark = '[x]' if (j in selected[cur]) else '[ ]'
                arrow = '>' if (j == sel_hunk_idx) else ' '
                stdscr.addstr(3 + j, left_w, f"{arrow} {mark} {tag} old:{i1}-{i2} new:{j1}-{j2}")
            # Security overlay
            if show_security:
                stdscr.addstr(height - 8, 0, '-' * (width - 1))
                # Render header
                hdr = last_sec_header or security_report[:2]
                for idx, ln in enumerate(hdr[:2]):
                    stdscr.addstr(height - 7 + idx, 0, ln[: width - 1])
                # Render findings (optionally JS-only)
                def _is_js(f):
                    try:
                        fp = str(f.get('filename') or '')
                        if fp.endswith(('.js','.jsx','.ts','.tsx','.mjs','.cjs')):
                            return True
                        tool = str(f.get('tool') or '').lower()
                        return tool in {'eslint','tsc'}
                    except Exception:
                        return False
                def _tool_ok(f):
                    t = str(f.get('tool') or '').lower()
                    if show_eslint_only and t != 'eslint':
                        return False
                    if show_tsc_only and t != 'tsc':
                        return False
                    return True
                # File badge/selector
                cur_file = sec_file_list[sec_file_idx] if sec_file_list else None
                if cur_file:
                    badges = sec_file_counts.get(cur_file, {})
                    badge_s = f"eslint:{badges.get('eslint',0)} tsc:{badges.get('tsc',0)} other:{badges.get('other',0)}"
                    stdscr.addstr(height - 5, 0, f"File {sec_file_idx+1}/{len(sec_file_list)}: {cur_file}  [{badge_s}]"[: width - 1])
                rows = []
                shown = 0
                for f in last_sec_findings:
                    if show_js_only and not _is_js(f):
                        continue
                    if not _tool_ok(f):
                        continue
                    if cur_file and str(f.get('filename') or '') != cur_file:
                        continue
                    rows.append(f" - [{f.get('severity')}] {f.get('rule_id')} {f.get('filename')}:{f.get('line')} — {f.get('message')}")
                    shown += 1
                    if shown >= 5:
                        break
                if not rows:
                    note = "(no JS/TS findings)" if show_js_only else "(no findings)"
                    stdscr.addstr(height - 4, 0, note[: width - 1])
                else:
                    for idx2, ln in enumerate(rows):
                        stdscr.addstr(height - 4 - (len(rows)-1) + idx2, 0, ln[: width - 1])
            # Tests overlay
            if show_tests:
                stdscr.addstr(height - 10, 0, '-' * (width - 1))
                for idx, ln in enumerate(tests_report[:4]):
                    stdscr.addstr(height - 9 + idx, 0, ln[: width - 1])
            # JS affected overlay
            if show_js_aff:
                stdscr.addstr(height - 14, 0, '-' * (width - 1))
                stdscr.addstr(height - 13, 0, '[JS] Affected importers (depth 1):')
                for idx, ln in enumerate(js_aff_lines[:6]):
                    stdscr.addstr(height - 12 + idx, 0, ln[: width - 1])
            # Python rename affected overlay
            if show_py_aff:
                stdscr.addstr(height - 20, 0, '-' * (width - 1))
                stdscr.addstr(height - 19, 0, '[Py] Rename affected (depth 2):')
                for idx, ln in enumerate(py_aff_lines[:6]):
                    stdscr.addstr(height - 18 + idx, 0, ln[: width - 1])
            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (ord('q'), 27):
                break
            elif ch in (ord('j'), curses.KEY_DOWN):
                sel_file_idx = min(sel_file_idx + 1, max(0, len(files) - 1))
                sel_hunk_idx = 0
            elif ch in (ord('k'), curses.KEY_UP):
                sel_file_idx = max(sel_file_idx - 1, 0)
                sel_hunk_idx = 0
            elif ch in (ord('l'), curses.KEY_RIGHT):
                sel_hunk_idx = min(sel_hunk_idx + 1, max(0, len(per_file_hunks.get(files[sel_file_idx], [])) - 1))
            elif ch in (ord('h'), curses.KEY_LEFT):
                sel_hunk_idx = max(sel_hunk_idx - 1, 0)
            elif ch == ord(' '):
                if per_file_hunks.get(files[sel_file_idx]):
                    if sel_hunk_idx in selected[files[sel_file_idx]]:
                        selected[files[sel_file_idx]].remove(sel_hunk_idx)
                    else:
                        selected[files[sel_file_idx]].add(sel_hunk_idx)
            elif ch == ord('g'):
                show_security = True
                _compute_security()
            elif ch in (ord('J')):
                # Toggle JS-only view for security overlay
                show_js_only = not show_js_only
            elif ch == ord('E'):
                # ESLint-only
                show_eslint_only = not show_eslint_only
                if show_eslint_only:
                    show_tsc_only = False
            elif ch == ord('T'):
                # tsc-only
                show_tsc_only = not show_tsc_only
                if show_tsc_only:
                    show_eslint_only = False
            elif ch == ord(']'):
                if sec_file_list:
                    sec_file_idx = (sec_file_idx + 1) % len(sec_file_list)
            elif ch == ord('['):
                if sec_file_list:
                    sec_file_idx = (sec_file_idx - 1) % len(sec_file_list)
            elif ch == ord('t'):
                pred = _predicted_from_selection()
                sig = _sig_predicted(pred)
                ok, failing, lines = _run_pytest_subset_in_shadow(pred)
                last_test_sig = sig
                last_test_ok = ok
                tests_report = lines[:50]
                show_tests = True
            elif ch == ord('I'):
                # Compute JS affected importers for current selection
                _compute_js_affected()
                show_js_aff = True
            elif ch == ord('R'):
                # Compute Python rename affected (if plan contains rename_symbol)
                _compute_py_affected()
                show_py_aff = True
            elif ch == ord('a'):
                pred = _predicted_from_selection()
                proceed, lines = _assess_security(pred)
                if not proceed:
                    show_security = True
                    security_report = ["[BLOCK] Security gate prevents apply:"] + lines
                    continue
                # Auto-run subset unless we already have a matching pass
                sig = _sig_predicted(pred)
                need_run = (last_test_sig != sig) or (last_test_ok is None)
                if need_run:
                    ok, failing, tlines = _run_pytest_subset_in_shadow(pred)
                    last_test_sig = sig
                    last_test_ok = ok
                    tests_report = tlines[:50]
                    show_tests = True
                if last_test_ok is False:
                    if not show_security:
                        show_security = True
                    security_report = ["[BLOCK] Test subset failed — press 'A' to force apply or adjust selection."]
                    continue
                _apply_selection()
                break
            elif ch == ord('A'):
                _apply_selection()
                break

    curses.wrapper(_draw)
    # Save prefs on exit
    try:
        if 'save_prefs' in globals() and callable(save_prefs):
            new = dict(_ov)
            new.update({
                'show_security': bool(show_security),
                'js_only': bool(show_js_only),
                'eslint_only': bool(show_eslint_only),
                'tsc_only': bool(show_tsc_only),
                'last_profile': (os.getenv('NERION_POLICY') or 'balanced'),
            })
            prefs = dict(_prefs or {}) if '_prefs' in globals() else {}
            prefs['overlay'] = new
            # Persist last test subset nodeids if set via env
            subset_env = (os.getenv('NERION_PATCH_TUI_NODEIDS') or '').strip()
            if subset_env:
                prefs['last_nodeids'] = subset_env
            save_prefs(prefs)
    except Exception:
        pass
    return 0


# ---------------- Safe Apply (non-TUI) -----------------

def _write_predicted_into_shadow_simple(shadow_root: Path, predicted: Dict[str, str], repo_root: Path) -> None:
    for fname, text in (predicted or {}).items():
        try:
            p = Path(fname)
            rel = p
            if p.is_absolute():
                try:
                    rel = p.relative_to(repo_root)
                except Exception:
                    rel = Path(p.name)
            dest = shadow_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text or '', encoding='utf-8')
        except Exception:
            continue


def _run_pytest_subset_in_shadow_simple(predicted: Dict[str, str]) -> Tuple[bool, List[str]]:
    from selfcoder.simulation import make_shadow_copy
    try:
        from ops.security.safe_subprocess import safe_run
    except Exception:
        import subprocess
        def safe_run(argv, **kwargs):
            return subprocess.run(argv, **{k: v for k, v in kwargs.items() if k in ("cwd","timeout","check","capture_output","text")})
    root = Path('.').resolve()
    shadow = make_shadow_copy(root)
    try:
        _write_predicted_into_shadow_simple(shadow, predicted, root)
        res = safe_run([os.sys.executable, '-m', 'pytest', '-q'], cwd=shadow, timeout=300, check=False, capture_output=True, text=True)
        ok = (res.returncode == 0 or res.returncode == 5)
        lines = (res.stdout or '').splitlines()[-10:]
        return ok, lines
    finally:
        try:
            import shutil as _shutil
            _shutil.rmtree(shadow, ignore_errors=True)
        except Exception:
            pass


def _predict_all_for_plan(planfile: Path, only_files: Optional[List[str]] = None) -> Tuple[Dict[str, str], List[Path], List[dict]]:
    from selfcoder.orchestrator import _apply_actions_preview as preview
    plan = json.loads(planfile.read_text(encoding='utf-8'))
    actions = plan.get('actions') or []
    if only_files:
        files = [Path(f) for f in only_files]
    elif plan.get('files'):
        files = [Path(p) for p in plan['files']]
    elif plan.get('target_file'):
        files = [Path(plan['target_file'])]
    else:
        files = []
    pv = preview(files, actions)
    predicted: Dict[str, str] = {str(p): pair[1] for p, pair in pv.items()}
    return predicted, files, actions


def cmd_safe_apply(args: argparse.Namespace) -> int:
    from core.ui.messages import fmt, Result
    from core.ui.progress import progress
    from selfcoder.reviewers.reviewer import review_predicted_changes
    from selfcoder.orchestrator import run_actions_on_files as apply_files

    planfile = Path(args.planfile)
    only_files = [str(f) for f in (getattr(args, 'file', []) or [])]

    with progress("Safe-apply: predict"):
        predicted, files, actions = _predict_all_for_plan(planfile, only_files or None)
        if not predicted:
            print(fmt('patch', 'safe-apply', Result.SKIP, 'no predicted changes'))
            return 1

    with progress("Safe-apply: review"):
        rep = review_predicted_changes(predicted, Path('.').resolve())
        sec = rep.get('security', {}) if isinstance(rep, dict) else {}
        proceed = bool(sec.get('proceed'))
        score = int(sec.get('score') or 0)
        if not proceed:
            msg = fmt('patch', 'safe-apply', Result.BLOCKED, f'security score={score}')
            print(msg)
            if getattr(args, 'json', False):
                print(json.dumps({'ok': False, 'blocked': 'security', 'score': score}, indent=2))
            return 1

    with progress("Safe-apply: tests"):
        ok, lines = _run_pytest_subset_in_shadow_simple(predicted)
        if not ok:
            msg = fmt('patch', 'safe-apply', Result.BLOCKED, 'tests failed')
            print(msg)
            if getattr(args, 'json', False):
                print(json.dumps({'ok': False, 'blocked': 'tests', 'lines': lines[-5:]}, indent=2))
            return 1

    with progress("Safe-apply: apply"):
        changed = apply_files(files, actions, dry_run=False)
        ok = bool(changed)
        msg = fmt('patch', 'safe-apply', Result.OK if ok else Result.SKIP, f"applied {len(changed)} file(s)")
        print(msg)
        if getattr(args, 'json', False):
            print(json.dumps({'ok': ok, 'changed': [str(p) for p in changed]}, indent=2))
        return 0 if ok else 1
