"""Local Reviewer agent (static checks and style hints).

Given a mapping of filename->new_source (predicted changes), runs security
gate checks and a small set of style hints to produce a human-friendly report.
Offline only; no model calls.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import os
import sys
import json

from selfcoder.security.gate import assess_plan
from core.ui.messages import fmt as _fmt_msg
from core.ui.messages import Result as _MsgRes
try:
    from selfcoder.security.policy_messages import explain_policy as _explain_policy
except Exception:
    _explain_policy = None  # graceful fallback
from ops.security.safe_subprocess import safe_run
import ast


def _style_hints(filename: str, text: str) -> List[str]:
    hints: List[str] = []
    # Long lines
    for i, ln in enumerate(text.splitlines(), 1):
        try:
            if len(ln) > 100:
                hints.append(f"{filename}:{i} line >100 chars")
        except Exception:
            continue
    # Module docstring
    if text and not text.lstrip().startswith(("\"\"\"", "'''")):
        hints.append(f"{filename}: consider adding a module docstring")
    # Import hygiene
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.names and any(getattr(n, 'name', '') == '*' for n in node.names):
                hints.append(f"{filename}:{getattr(node,'lineno',1)} avoid wildcard import '*'")
    except Exception:
        pass
    return hints


def review_predicted_changes(predicted: Dict[str, str], repo_root: Path) -> Dict[str, Any]:
    """Return a structured review report for predicted file changes."""
    report: Dict[str, Any] = {
        "security": {},
        "style": {},
        "summary": {},
    }
    # Security assessment (aggregated)
    gate = assess_plan(predicted, repo_root)
    report["security"] = {
        "proceed": bool(gate.proceed),
        "score": int(gate.score),
        "reason": gate.reason,
        "findings": [
            {
                "rule_id": f.rule_id,
                "severity": f.severity,
                "filename": f.filename,
                "line": f.line,
                "message": f.message,
            }
            for f in (gate.findings or [])
        ],
    }
    # Style/complexity hints per file
    style: Dict[str, List[str]] = {}
    for fname, new_src in predicted.items():
        file_hints = _style_hints(fname, new_src or "")
        # Cyclomatic complexity (very rough): count branching keywords
        try:
            src = new_src or ""
            complexity = sum(src.count(k) for k in (" if ", " for ", " while ", " and ", " or "))
            if complexity > 30:
                file_hints.append(f"{fname}: high branching complexity ({complexity}) — consider refactor")
        except Exception:
            pass
        style[fname] = file_hints
    report["style"] = style
    # External linters/type-checkers (optional)
    external: Dict[str, Any] = {}
    try:
        tmp_root = (repo_root / '.nerion' / 'review_tmp')
        tmp_root.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for fname, new_src in predicted.items():
            try:
                p = Path(fname)
                # Normalize to a relative path under tmp_root to avoid writing real files
                rel = p
                if rel.is_absolute():
                    try:
                        # Drop root (first component) to keep structure shallow
                        rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)
                    except Exception:
                        rel = Path(p.name)
                dest = tmp_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(new_src or '', encoding='utf-8')
                written.append(dest)
            except Exception:
                continue
        # Policy: in fast mode, skip external linters entirely
        try:
            from selfcoder.config import get_policy as _get_policy
            _policy = _get_policy()
        except Exception:
            _policy = 'balanced'
        if _policy == 'fast':
            external['skipped_by_policy'] = True
        # Ruff
        if _policy != 'fast' and (os.getenv('NERION_REVIEW_RUFF') or '').strip().lower() in {'1','true','yes','on'} and written:
            try:
                res = safe_run([sys.executable, '-m', 'ruff', 'check', '--quiet', '--format', 'json', *[str(p) for p in written]], check=False, capture_output=True, text=True)
                issues = []
                try:
                    issues = json.loads(res.stdout or '[]')
                except Exception:
                    issues = []
                external['ruff'] = {'count': len(issues)}
            except Exception:
                pass
        # pydocstyle
        if _policy != 'fast' and (os.getenv('NERION_REVIEW_PYDOCSTYLE') or '').strip().lower() in {'1','true','yes','on'} and written:
            try:
                res = safe_run([sys.executable, '-m', 'pydocstyle', *[str(p) for p in written]], check=False, capture_output=True, text=True)
                count = 0
                if res.returncode != 0 and res.stdout:
                    count = sum(1 for ln in res.stdout.splitlines() if ln.strip().startswith(str(tmp_root)))
                external['pydocstyle'] = {'count': int(count)}
            except Exception:
                pass
        # mypy (best-effort)
        if _policy != 'fast' and (os.getenv('NERION_REVIEW_MYPY') or '').strip().lower() in {'1','true','yes','on'} and written:
            try:
                res = safe_run([sys.executable, '-m', 'mypy', '--no-color-output', '--hide-error-context', '--ignore-missing-imports', *[str(p) for p in written]], check=False, capture_output=True, text=True)
                count = 0
                if res.returncode != 0 and res.stdout:
                    count = sum(1 for ln in res.stdout.splitlines() if ': error:' in ln)
                external['mypy'] = {'count': int(count)}
            except Exception:
                pass
    except Exception:
        pass

    # Summary
    total_hints = sum(len(v) for v in style.values())
    report["summary"] = {
        "files": len(predicted),
        "security_ok": bool(gate.proceed),
        "security_findings": len(gate.findings or []),
        "style_hints": int(total_hints),
    }
    if external:
        report['external'] = external
    return report


def format_review(report: Dict[str, Any]) -> str:
    parts: List[str] = []
    sec = report.get("security", {})
    # Unified top-line message
    top_detail = f"score={sec.get('score', 0)}"
    if sec.get('reason'):
        top_detail = f"{top_detail}; {sec.get('reason')}"
    parts.append(_fmt_msg("review", "security", _MsgRes.OK if sec.get('proceed') else _MsgRes.BLOCKED, top_detail))
    for f in sec.get("findings", [])[:20]:
        parts.append(f" - [{f.get('severity')}] {f.get('rule_id')} {f.get('filename')}:{f.get('line')} — {f.get('message')}")
        # Add friendly policy explanation lines
        try:
            rid = str(f.get('rule_id') or '')
            if _explain_policy and rid.startswith('POLICY:'):
                kind = 'paths'
                if rid == 'POLICY:ACTION':
                    kind = 'actions'
                elif rid == 'POLICY:LIMIT':
                    kind = 'size'
                elif rid == 'POLICY:PATH':
                    kind = 'paths'
                block = {
                    'kind': kind,
                    'rule': rid.split(':',1)[1].lower() if ':' in rid else 'policy',
                    'value': '',
                    'path': f.get('filename') or '',
                    'action': '',
                }
                ex = _explain_policy(block)
                parts.append(f"   • Why: {ex.get('reason')}")
                parts.append(f"   • Fix: {ex.get('hint')} ({ex.get('doc')})")
        except Exception:
            pass
    style = report.get("style", {})
    if any(style.values()):
        parts.append("Style hints:")
        for fname, hints in style.items():
            for h in hints[:20]:
                parts.append(f" - {h}")
    s = report.get("summary", {})
    parts.append(
        f"Summary: files={s.get('files',0)} findings={s.get('security_findings',0)} hints={s.get('style_hints',0)}"
    )
    return "\n".join(parts)
