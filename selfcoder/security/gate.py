from pathlib import Path
from typing import Dict, List, Any
from . import GateResult, Finding
from .scanner import scan_source
from .extlinters import run_on_dir as _run_ext

SEVERITY_SCORE = {"low": 1, "medium": 3, "high": 7, "critical": 10}


def _policy_thresholds() -> int:
    try:
        from selfcoder.config import get_policy as _get_policy
        pol = _get_policy('balanced')
    except Exception:
        pol = 'balanced'
    if pol == 'safe':
        return 8
    if pol == 'fast':
        return 999999
    return 10


def assess_plan(predicted_changes: Dict[str, str], repo_root: Path,
                block_on_critical: bool = True,
                block_score_threshold: int | None = None,
                plan_actions: List[Dict[str, Any]] | None = None) -> GateResult:
    all_findings: List[Finding] = []
    score = 0
    has_critical = False
    # Internal AST/regex scanner on predicted content
    for fname, new_src in predicted_changes.items():
        all_findings.extend(scan_source(new_src, fname, repo_root))
    # External linters on a temp tree (best-effort; skipped in fast policy)
    try:
        tmp_root = (repo_root / '.nerion' / 'review_tmp')
        tmp_root.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for fname, new_src in predicted_changes.items():
            p = Path(fname)
            rel = p
            if rel.is_absolute():
                try:
                    rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)
                except Exception:
                    rel = Path(p.name)
            dest = tmp_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(new_src or '', encoding='utf-8')
            written.append(dest)
        ext = _run_ext(tmp_root, [w.relative_to(tmp_root) for w in written])
        for e in ext:
            sev = str(e.get('severity') or 'low').lower()
            all_findings.append(Finding(rule_id=f"{e.get('tool')}:{e.get('code')}",
                                       severity=sev, message=str(e.get('message') or ''),
                                       filename=str(e.get('path') or ''), line=int(e.get('line') or 0),
                                       evidence=''))
    except Exception:
        pass
    # Policy enforcement (paths/limits/actions)
    policy_block = False
    try:
        from selfcoder.security.policy import load_policy as _load_pol, enforce_paths as _enf_paths, enforce_limits as _enf_limits, enforce_actions as _enf_acts
        import os as _os
        if (_os.getenv('NERION_DISABLE_POLICY') or '').strip() == '1' and (_os.getenv('CI') or '').strip().lower() in {'1','true','yes','on'}:
            print('[policy] DISABLED (CI): Policy enforcement bypassed — ensure this is only used in CI')
            pol = {}
        else:
            pol = _load_pol(repo_root)
        paths = []
        for k in predicted_changes.keys():
            p = Path(k)
            try:
                rel = p.relative_to(repo_root)
            except Exception:
                rel = p
            paths.append(rel)
        okp, why, viol = _enf_paths(paths, pol)
        if not okp:
            for v in viol or []:
                all_findings.append(Finding(rule_id='POLICY:PATH', severity='high', message=why, filename=v.as_posix(), line=0, evidence=''))
            policy_block = True
        okL, whyL = _enf_limits(predicted_changes, pol)
        if not okL:
            all_findings.append(Finding(rule_id='POLICY:LIMIT', severity='high', message=whyL, filename='', line=0, evidence=''))
            policy_block = True
        if plan_actions:
            okA, whyA = _enf_acts(plan_actions, pol)
            if not okA:
                all_findings.append(Finding(rule_id='POLICY:ACTION', severity='high', message=whyA, filename='', line=0, evidence=''))
                policy_block = True
    except Exception:
        pass

    # Score and thresholds
    for f in all_findings:
        score += SEVERITY_SCORE.get(f.severity, 0)
        if f.severity == "critical":
            has_critical = True
    proceed = True
    reason = "no issues detected"
    if block_on_critical and has_critical:
        proceed = False
        reason = "critical finding(s) present"
    if policy_block:
        proceed = False
        reason = 'policy violation'
    thresh = block_score_threshold if block_score_threshold is not None else _policy_thresholds()
    if proceed and score >= int(thresh):
        proceed = False
        reason = f"risk score {score} ≥ threshold {thresh}"
    return GateResult(proceed=proceed, score=score, findings=all_findings, reason=reason)
