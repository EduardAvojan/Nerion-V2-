from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from selfcoder.reviewers.reviewer import review_predicted_changes, format_review
from selfcoder.tester.expander import expand_tests
from ops.security import fs_guard
from core.ui.progress import progress


def cmd_preflight(args: argparse.Namespace) -> int:
    planfile = Path(getattr(args, 'planfile'))
    data: Dict[str, Any] = json.loads(planfile.read_text(encoding='utf-8'))
    actions = data.get('actions') or []
    targets_cli = [Path(f) for f in (getattr(args, 'file', []) or [])]
    target = data.get('target_file') or (data.get('files') or [None])[0]
    if not (targets_cli or target) or not actions:
        print('[preflight] plan missing target_file or actions')
        return 1

    # Predicted preview
    try:
        from selfcoder.orchestrator import _apply_actions_preview as preview
    except Exception:
        print('[preflight] preview unavailable')
        return 1
    files = targets_cli or [Path(target)]
    with progress("Preflight: preview"):
        previews = preview(files, actions)
    predicted = {p.as_posix(): new for p, (_old, new) in previews.items()}

    # Reviewer
    with progress("Preflight: review"):
        rep = review_predicted_changes(predicted, fs_guard.infer_repo_root(Path('.')))
    # Tester expansion (text only; does not write files)
    with progress("Preflight: tester"):
        edge_code = expand_tests(data, target)

    if getattr(args, 'json', False):
        out = {
            'review': rep,
            'tester': {'edge_tests_snippet': edge_code[:800]},
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print('[review]')
    print(format_review(rep))
    try:
        from core.ui.messages import fmt as _fmt_msg, Result as _MsgRes
        sec = rep.get('security', {})
        print(_fmt_msg('preflight', 'summary', _MsgRes.OK if sec.get('proceed') else _MsgRes.BLOCKED, f"score={sec.get('score',0)} findings={len(sec.get('findings') or [])}"))
    except Exception:
        pass
    # Fix hints block: suggest useful flags and scaffolds
    total_hints = sum(len(v or []) for v in (rep.get('style') or {}).values())
    fixes: list[str] = []
    if total_hints:
        fixes.append("Consider enabling external linters for this preview: set NERION_REVIEW_RUFF=1, NERION_REVIEW_PYDOCSTYLE=1, NERION_REVIEW_MYPY=1")
    # Suggest docstring scaffolds when module docstring hints present
    if any('module docstring' in h for hints in (rep.get('style') or {}).values() for h in hints):
        fixes.append("Add module/function docstrings via: nerion docstring --help")
    if fixes:
        print('\n[fix hints]')
        for f in fixes:
            print(f" - {f}")
    print('\n[tester] (edge-case test stubs preview)\n')
    print(edge_code)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('preflight', help='run Reviewer + Tester preview on a plan (no writes)')
    p.add_argument('planfile', help='path to plan JSON')
    p.add_argument('--file', action='append', default=[], help='limit review/tester preview to selected files (repeatable)')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_preflight)
