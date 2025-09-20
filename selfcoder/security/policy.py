"""Policy DSL loader and evaluators.

Policy locations (first found wins):
  - .nerion/policy.yaml
  - config/policy.yaml

Schema (keys optional):
  actions:
    allow: [insert_function, add_module_docstring]
    deny: [rename_symbol]
  paths:
    allow: ["selfcoder/**", "app/**"]
    deny: ["plugins/**"]
  limits:
    max_file_bytes: 200000
    max_total_bytes: 500000
  secrets:
    block: true  # block on any secret finding (scanner)
  network:
    block_requests: true  # treat requests.* as high severity
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import fnmatch


def _load_yaml(p: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _normalize(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both flat and wrapped forms; normalize synonyms to internal shape.

    Internal shape:
      actions: {allow: [...], deny: [...]}
      paths:   {allow: [...], deny: [...]}
      limits:  {max_file_bytes, max_total_bytes, max_files}
      secrets: {block: bool}
      network: {block_requests: bool}
    """
    data = schema.get('policy') if isinstance(schema.get('policy'), dict) else schema
    out: Dict[str, Any] = {}
    # actions
    acts = data.get('actions') or {}
    if 'allow_actions' in data or 'deny_actions' in data:
        acts = {
            'allow': list(data.get('allow_actions') or []),
            'deny': list(data.get('deny_actions') or []),
        }
    out['actions'] = {
        'allow': [str(x) for x in (acts.get('allow') or []) if isinstance(x, str)],
        'deny': [str(x) for x in (acts.get('deny') or []) if isinstance(x, str)],
    }
    # paths
    paths = data.get('paths') or {}
    if 'allow_paths' in data or 'deny_paths' in data:
        paths = {
            'allow': list(data.get('allow_paths') or []),
            'deny': list(data.get('deny_paths') or []),
        }
    out['paths'] = {
        'allow': [str(x) for x in (paths.get('allow') or []) if isinstance(x, str)],
        'deny': [str(x) for x in (paths.get('deny') or []) if isinstance(x, str)],
    }
    # limits
    limits = data.get('limits') or {}
    for k_syn, k_norm in (('max_diff_bytes','max_total_bytes'),):
        if k_syn in data and k_norm not in limits:
            limits[k_norm] = data[k_syn]
    out['limits'] = {
        'max_file_bytes': int(limits.get('max_file_bytes') or 0),
        'max_total_bytes': int(limits.get('max_total_bytes') or 0),
        'max_files': int(limits.get('max_files') or 0),
    }
    # secrets/network
    out['secrets'] = {'block': bool((data.get('secrets_block') if 'secrets_block' in data else (data.get('secrets') or {}).get('block')) or False)}
    out['network'] = {'block_requests': bool((data.get('block_requests') if 'block_requests' in data else (data.get('network') or {}).get('block_requests')) or False)}
    return out


def load_policy(repo_root: Path, *, explicit_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and normalize a policy file.

    Precedence:
      1) explicit_path if provided
      2) NERION_POLICY_FILE env if set
      3) .nerion/policy.yaml, config/policy.yaml (first found)
    """
    import os
    root = Path(repo_root)
    # Allow env override
    env_path = os.getenv('NERION_POLICY_FILE')
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([root / '.nerion' / 'policy.yaml', root / 'config' / 'policy.yaml'])
    for p in candidates:
        data = _load_yaml(Path(p))
        if data:
            return _normalize(data)
    return {}


def _match_any(path: Path, patterns: List[str]) -> bool:
    s = path.as_posix()
    for pat in patterns or []:
        try:
            if fnmatch.fnmatch(s, pat):
                return True
        except Exception:
            continue
    return False


def enforce_actions(plan_actions: List[Dict[str, Any]] | None, policy: Dict[str, Any]) -> Tuple[bool, str]:
    acts = plan_actions or []
    pol = policy.get('actions') or {}
    allow = [str(a) for a in (pol.get('allow') or []) if isinstance(a, str)]
    deny = [str(a) for a in (pol.get('deny') or []) if isinstance(a, str)]
    if not allow and not deny:
        return True, ''
    for a in acts:
        try:
            k = (a.get('kind') or a.get('action') or '').strip()
        except Exception:
            k = ''
        if not k:
            continue
        if deny and k in deny:
            return False, f"action '{k}' is denied by policy"
        if allow and (k not in allow):
            return False, f"action '{k}' not in allow-list"
    return True, ''


def enforce_paths(predicted_paths: List[Path], policy: Dict[str, Any]) -> Tuple[bool, str, List[Path]]:
    pol = policy.get('paths') or {}
    allow = [str(a) for a in (pol.get('allow') or []) if isinstance(a, str)]
    deny = [str(a) for a in (pol.get('deny') or []) if isinstance(a, str)]
    violations: List[Path] = []
    for p in predicted_paths or []:
        if deny and _match_any(p, deny):
            violations.append(p)
            continue
        if allow and (not _match_any(p, allow)):
            violations.append(p)
    if violations:
        if deny and any(_match_any(v, deny) for v in violations):
            return False, 'denied path pattern matched', violations
        return False, 'path not allowed by allow-list', violations
    return True, '', []


def enforce_limits(predicted: Dict[str, str], policy: Dict[str, Any]) -> Tuple[bool, str]:
    limits = policy.get('limits') or {}
    try:
        max_file = int(limits.get('max_file_bytes') or 0)
    except Exception:
        max_file = 0
    try:
        max_total = int(limits.get('max_total_bytes') or 0)
    except Exception:
        max_total = 0
    try:
        max_files = int(limits.get('max_files') or 0)
    except Exception:
        max_files = 0
    if not max_file and not max_total:
        return True, ''
    total = 0
    for fname, text in (predicted or {}).items():
        b = len((text or '').encode('utf-8'))
        total += b
        if max_file and b > max_file:
            return False, f"file {fname} exceeds max_file_bytes ({b}>{max_file})"
    if max_total and total > max_total:
        return False, f"total change exceeds max_total_bytes ({total}>{max_total})"
    if max_files and len(predicted or {}) > max_files:
        return False, f"number of files exceeds max_files ({len(predicted)}>{max_files})"
    return True, ''
