"""External linters/type-checkers aggregator (best-effort, offline).

Runs available tools over a temporary tree or real paths and returns a
normalized list of Finding-like dicts: {tool, code, message, path, line, severity}.

Tools: ruff, mypy, bandit, semgrep (optional). Missing tools are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import os
import sys

def _run(cmd: List[str], cwd: Path | None = None) -> Tuple[int, str, str]:
    try:
        import subprocess
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=False)
        return p.returncode, p.stdout or '', p.stderr or ''
    except Exception as e:
        return 1, '', str(e)


def _normalize_sev(tool: str, raw: str) -> str:
    s = (raw or '').lower()
    if tool == 'bandit':
        # bandit uses LOW/MEDIUM/HIGH
        if 'high' in s:
            return 'high'
        if 'medium' in s:
            return 'medium'
        return 'low'
    if tool == 'semgrep':
        # semgrep severities: ERROR/WARNING/INFO
        if 'error' in s:
            return 'high'
        if 'warn' in s:
            return 'medium'
        return 'low'
    if tool == 'mypy':
        # Treat type errors as medium
        return 'medium'
    # ruff default to low (style)
    return 'low'


def run_on_dir(root: Path, rel_paths: List[Path] | None = None) -> List[Dict[str, Any]]:
    root = Path(root)
    files = [p for p in (rel_paths or []) if (root / p).exists()]
    out: List[Dict[str, Any]] = []
    policy = (os.getenv('NERION_POLICY') or 'balanced').strip().lower()
    if policy == 'fast':
        return out

    # ruff
    try:
        cmd = [sys.executable, '-m', 'ruff', 'check', '--format', 'json'] + ([str(p) for p in files] if files else ['.'])
        rc, so, _ = _run(cmd, cwd=root)
        if so.strip():
            issues = json.loads(so)
            for it in issues or []:
                path = it.get('filename') or it.get('file') or ''
                line = int(((it.get('location') or {}).get('row')) or it.get('line') or 0)
                code = it.get('code') or it.get('rule') or 'RUF'
                msg = it.get('message') or ''
                out.append({'tool': 'ruff', 'code': code, 'message': msg, 'path': path, 'line': line, 'severity': _normalize_sev('ruff', '')})
    except Exception:
        pass

    # mypy (best-effort JSON format; fallback to text)
    try:
        cmd = [sys.executable, '-m', 'mypy', '--hide-error-context', '--no-error-summary', '--ignore-missing-imports', '--no-color-output', '--error-format=json'] + ([str(p) for p in files] if files else ['.'])
        rc, so, _ = _run(cmd, cwd=root)
        items = []
        try:
            items = json.loads(so or '[]')
        except Exception:
            items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get('type') != 'error':
                continue
            path = it.get('filename') or ''
            line = int(it.get('line') or 0)
            msg = it.get('message') or ''
            code = (it.get('code') or 'mypy')
            out.append({'tool': 'mypy', 'code': code, 'message': msg, 'path': path, 'line': line, 'severity': _normalize_sev('mypy', '')})
    except Exception:
        pass

    # bandit
    try:
        cmd = ['bandit', '-r'] + ([str(p) for p in files] if files else ['.']) + ['-f', 'json']
        rc, so, _ = _run(cmd, cwd=root)
        data = json.loads(so or '{}')
        for it in (data.get('results') or []):
            path = it.get('filename') or ''
            line = int(it.get('line_number') or 0)
            sev = str(it.get('issue_severity') or 'LOW')
            code = it.get('test_id') or 'B'
            msg = it.get('issue_text') or ''
            out.append({'tool': 'bandit', 'code': code, 'message': msg, 'path': path, 'line': line, 'severity': _normalize_sev('bandit', sev)})
    except Exception:
        pass

    # semgrep (optional)
    try:
        if (os.getenv('NERION_SEMGREP') or '').strip().lower() in {'1','true','yes','on'}:
            cmd = ['semgrep', 'scan', '--config', 'auto', '--json'] + ([str(p) for p in files] if files else ['.'])
            rc, so, _ = _run(cmd, cwd=root)
            data = json.loads(so or '{}')
            for it in (data.get('results') or []):
                extra = it.get('extra') or {}
                sev = extra.get('severity') or 'INFO'
                msg = extra.get('message') or ''
                path = ((it.get('path') or '') or '')
                line = int(((it.get('start') or {}).get('line')) or 0)
                code = (extra.get('rule') or 'SEMGREP')
                out.append({'tool': 'semgrep', 'code': code, 'message': msg, 'path': path, 'line': line, 'severity': _normalize_sev('semgrep', sev)})
    except Exception:
        pass

    # semgrep JS-only (opt-in)
    try:
        enable_js = (os.getenv('NERION_SEMGREP_JS') or '').strip().lower() in {'1','true','yes','on'}
        if enable_js:
            js_files = [p for p in (files or []) if str(p).endswith(('.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'))]
            if js_files:
                cmd = ['semgrep', 'scan', '--config', 'auto', '--json'] + [str(p) for p in js_files]
                rc, so, _ = _run(cmd, cwd=root)
                data = json.loads(so or '{}')
                for it in (data.get('results') or []):
                    extra = it.get('extra') or {}
                    sev = extra.get('severity') or 'INFO'
                    msg = extra.get('message') or ''
                    path = ((it.get('path') or '') or '')
                    line = int(((it.get('start') or {}).get('line')) or 0)
                    code = (extra.get('rule') or 'SEMGREP')
                    out.append({'tool': 'semgrep', 'code': code, 'message': msg, 'path': path, 'line': line, 'severity': _normalize_sev('semgrep', sev)})
    except Exception:
        pass

    # --- JS/TS tools (optional) -----------------------------------------
    # ESLint: only if installed and JS/TS files present or explicitly enabled via env
    try:
        enable_eslint = (os.getenv('NERION_ESLINT') or '').strip().lower() in {'1','true','yes','on'}
        has_js = any(str(p).endswith(('.js', '.ts', '.tsx')) for p in (rel_paths or []))
        eslint_bin = None
        # Prefer local project binary under node_modules/.bin
        local_bin = (root / 'node_modules' / '.bin' / ('eslint.cmd' if os.name == 'nt' else 'eslint'))
        if local_bin.exists():
            eslint_bin = str(local_bin)
        else:
            import shutil as _sh
            eslint_bin = _sh.which('eslint')
        if eslint_bin and (enable_eslint or has_js):
            cmd = [eslint_bin, '-f', 'json'] + ([str(p) for p in rel_paths] if rel_paths else ['.'])
            rc, so, se = _run(cmd, cwd=root)
            try:
                items = json.loads(so or '[]')
            except Exception:
                items = []
            for file_res in items or []:
                fpath = file_res.get('filePath') or ''
                for m in (file_res.get('messages') or []):
                    sev = 'medium' if int(m.get('severity') or 1) >= 2 else 'low'
                    out.append({'tool': 'eslint', 'code': str(m.get('ruleId') or 'ESL'),'message': str(m.get('message') or ''), 'path': fpath, 'line': int(m.get('line') or 0), 'severity': sev})
    except Exception:
        pass

    # TypeScript compiler (noEmit): only if installed and TS files present or explicitly enabled
    try:
        enable_tsc = (os.getenv('NERION_TSC') or '').strip().lower() in {'1','true','yes','on'}
        has_ts = any(str(p).endswith(('.ts', '.tsx')) for p in (rel_paths or []))
        tsc_bin = None
        local_bin = (root / 'node_modules' / '.bin' / ('tsc.cmd' if os.name == 'nt' else 'tsc'))
        if local_bin.exists():
            tsc_bin = str(local_bin)
        else:
            import shutil as _sh
            tsc_bin = _sh.which('tsc')
        if tsc_bin and (enable_tsc or has_ts):
            rc, so, se = _run([tsc_bin, '--noEmit'], cwd=root)
            # tsc returns non-zero on errors; parse stdout/stderr lines for 'error'
            text = (so or '') + '\n' + (se or '')
            for ln in text.splitlines():
                if ' error ' in ln or ln.strip().startswith('error '):
                    # try to parse file:line
                    path = ''
                    line = 0
                    parts = ln.split('(')
                    if parts and ':' in parts[0]:
                        path = parts[0].strip()
                        try:
                            line = int(ln.split(')')[0].split(',')[0].split('(')[-1])
                        except Exception:
                            line = 0
                    out.append({'tool': 'tsc', 'code': 'TS', 'message': ln.strip(), 'path': path, 'line': line, 'severity': 'medium'})
    except Exception:
        pass

    return out


def summarize(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
    per_tool: Dict[str, int] = {}
    for f in findings or []:
        sev = str(f.get('severity') or 'low').lower()
        counts[sev] = counts.get(sev, 0) + 1
        per_tool[f.get('tool') or ''] = per_tool.get(f.get('tool') or '', 0) + 1
    score = counts.get('low', 0) * 1 + counts.get('medium', 0) * 3 + counts.get('high', 0) * 7 + counts.get('critical', 0) * 10
    return {'counts': counts, 'per_tool': per_tool, 'score': score}
