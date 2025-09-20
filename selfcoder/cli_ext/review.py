from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfcoder.reviewers.reviewer import review_predicted_changes, format_review
from core.ui.messages import fmt as _fmt_msg
from core.ui.messages import Result as _MsgRes
from core.ui.progress import progress
from ops.security import fs_guard


def cmd_review(args: argparse.Namespace) -> int:
    planfile = Path(getattr(args, 'planfile'))
    data = json.loads(planfile.read_text(encoding='utf-8'))
    actions = data.get('actions') or []
    targets_cli = [Path(f) for f in (getattr(args, 'file', []) or [])]
    target = data.get('target_file') or (data.get('files') or [None])[0]
    if not (targets_cli or target) or not actions:
        print('[review] plan missing target_file or actions')
        return 1
    # Build predicted mapping by previewing transforms
    try:
        from selfcoder.orchestrator import _apply_actions_preview as preview
    except Exception:
        print('[review] preview unavailable')
        return 1
    files = targets_cli or [Path(target)]
    with progress("Review: preview"):
        previews = preview(files, actions)
    predicted = {p.as_posix(): new for p, (_old, new) in previews.items()}
    with progress("Review: analyze"):
        rep = review_predicted_changes(predicted, fs_guard.infer_repo_root(Path('.')))
    if getattr(args, 'json', False):
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print(format_review(rep))
        # Final one-liner ack for scripts
        sec = rep.get('security', {})
        print(_fmt_msg('review', 'summary', _MsgRes.OK if sec.get('proceed') else _MsgRes.BLOCKED, f"score={sec.get('score',0)} findings={len(sec.get('findings') or [])}"))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('review', help='review a plan for security/style issues before applying')
    p.add_argument('planfile', help='path to plan JSON')
    p.add_argument('--file', action='append', default=[], help='limit review to selected files (repeatable)')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_review)
