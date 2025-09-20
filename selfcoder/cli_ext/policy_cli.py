from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load(repo: Path) -> Dict[str, Any]:
    try:
        from selfcoder.security.policy import load_policy
        return load_policy(repo)
    except Exception:
        return {}


def _audit(repo: Path) -> Dict[str, Any]:
    pol = _load(repo)
    out: Dict[str, Any] = {'policy': pol, 'violations': []}
    try:
        from selfcoder.security.policy import enforce_paths as _enf_paths, enforce_limits as _enf_limits
        # Collect all .py files as predicted for path/limit checks (dry run)
        preds: Dict[str, str] = {}
        for p in repo.rglob('*.py'):
            try:
                preds[str(p)] = p.read_text(encoding='utf-8')
            except Exception:
                continue
        okP, whyP, viol = _enf_paths([Path(k) for k in preds.keys()], pol)
        if not okP:
            out['violations'].append({'type': 'paths', 'reason': whyP, 'files': [v.as_posix() for v in viol]})
        okL, whyL = _enf_limits(preds, pol)
        if not okL:
            out['violations'].append({'type': 'limits', 'reason': whyL})
    except Exception as e:
        out['error'] = str(e)
    return out


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('policy', help='show/audit security policy (allow/deny/limits)')
    sp = p.add_subparsers(dest='policy_cmd', required=True)

    sh = sp.add_parser('show', help='print the merged policy for this repo')
    sh.add_argument('--policy', help='explicit policy file path (overrides env)')
    def _run_show(_args: argparse.Namespace) -> int:
        path = getattr(_args, 'policy', None)
        if path:
            from selfcoder.security.policy import load_policy
            pol = load_policy(Path('.').resolve(), explicit_path=Path(path))
        else:
            pol = _load(Path('.').resolve())
        print(json.dumps(pol or {}, indent=2))
        return 0
    sh.set_defaults(func=_run_show)

    au = sp.add_parser('audit', help='dry-run audit against policy: path patterns and limits')
    au.add_argument('--json', action='store_true')
    au.add_argument('--policy', help='explicit policy file path (overrides env)')
    def _run_audit(args: argparse.Namespace) -> int:
        if getattr(args, 'policy', None):
            from selfcoder.security.policy import load_policy
            repo = Path('.').resolve()
            pol = load_policy(repo, explicit_path=Path(getattr(args, 'policy')))
            # monkeypatch _load to use explicit
            def _load_explicit(_repo: Path):
                return pol
            globals()['_load'] = lambda _r: pol
        out = _audit(Path('.').resolve())
        if getattr(args, 'json', False):
            print(json.dumps(out, indent=2))
            return 0
        print('[policy] audit')
        if out.get('violations'):
            for v in out['violations']:
                t = v.get('type')
                r = v.get('reason')
                print(f" - {t}: {r}")
                if isinstance(v.get('files'), list):
                    for f in v.get('files')[:20]:
                        print(f"    • {f}")
        else:
            print('  OK')
        return 0
    au.set_defaults(func=_run_audit)
