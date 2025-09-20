from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from selfcoder.policy.profile_resolver import decide as _decide, _DEFAULT_PATH as _PROFILES_PATH
except Exception:
    _decide = None
    _PROFILES_PATH = Path("config/profiles.yaml")

try:
    from selfcoder.learning.continuous import load_prefs as _load_prefs, save_prefs as _save_prefs
except Exception:
    def _load_prefs():
        return {}
    def _save_prefs(p):
        Path('out/learning').mkdir(parents=True, exist_ok=True)
        Path('out/learning/prefs.json').write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding='utf-8')


def _show_env() -> dict:
    keys = [
        'NERION_POLICY',
        'NERION_REVIEW_STRICT', 'NERION_REVIEW_STYLE_MAX',
        'NERION_REVIEW_RUFF', 'NERION_REVIEW_PYDOCSTYLE', 'NERION_REVIEW_MYPY',
        'NERION_BENCH_USE_LIBPYTEST', 'NERION_BENCH_PYTEST_TIMEOUT',
        'NERION_CODER_BACKEND','NERION_CODER_MODEL','NERION_CODER_BASE_URL',
        'NERION_LLM_MODEL','NERION_ALLOW_NETWORK'
    ]
    return {k: os.getenv(k) for k in keys if os.getenv(k) is not None}


def cmd_show(_args: argparse.Namespace) -> int:
    out = {
        'profiles_yaml': str(_PROFILES_PATH) if _PROFILES_PATH.exists() else None,
        'env': _show_env(),
        'sticky_overrides': (_load_prefs().get('profile_overrides') or {}),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    task = getattr(args, 'task')
    sig = {}
    if getattr(args, 'signals', None):
        try:
            sig = json.loads(args.signals)
        except Exception:
            sig = {}
    if not _decide:
        print(json.dumps({'error': 'resolver unavailable'}, indent=2))
        return 1
    dec = _decide(task, signals=sig)
    print(json.dumps({'task': task, 'decision': {'name': dec.name, 'env': dec.env, 'why': dec.why}}, indent=2))
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    prefs = _load_prefs()
    ov = prefs.get('profile_overrides') or {}
    ov[str(args.task)] = str(args.profile)
    prefs['profile_overrides'] = ov
    _save_prefs(prefs)
    print(json.dumps({'set': {args.task: args.profile}}, indent=2))
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    prefs = _load_prefs()
    ov = prefs.get('profile_overrides') or {}
    if args.task in ov:
        ov.pop(args.task)
    prefs['profile_overrides'] = ov
    _save_prefs(prefs)
    print(json.dumps({'cleared': args.task}, indent=2))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('profile', help='profile resolver tools')
    sp = p.add_subparsers(dest='profile_cmd', required=True)

    sh = sp.add_parser('show', help='show current profile-related env and sticky overrides')
    sh.set_defaults(func=cmd_show)

    ex = sp.add_parser('explain', help='explain resolver decision for a task')
    ex.add_argument('--task', required=True)
    ex.add_argument('--signals', help='JSON dict with resolver signals')
    ex.set_defaults(func=cmd_explain)

    st = sp.add_parser('set', help='set sticky profile for a task')
    st.add_argument('--task', required=True)
    st.add_argument('--profile', required=True)
    st.set_defaults(func=cmd_set)

    cl = sp.add_parser('clear', help='clear sticky profile for a task')
    cl.add_argument('--task', required=True)
    cl.set_defaults(func=cmd_clear)
