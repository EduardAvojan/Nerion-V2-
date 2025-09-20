"""Optional Node.js ts-morph bridge for robust JS/TS transforms.

If Node or ts-morph runner is unavailable, returns None so callers can fall back
to textual transformers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import tempfile
import shutil
import subprocess


def _node_bin() -> Optional[str]:
    # Prefer system node
    try:
        return shutil.which('node')
    except Exception:
        return None


def _runner_path() -> Optional[Path]:
    here = Path(__file__).resolve().parents[2]  # repo root
    p = here / 'tools' / 'js' / 'tsmorph_runner.js'
    return p if p.exists() else None


def apply_actions_js_ts_node(source: str, actions: List[Dict[str, Any]], *, timeout_s: int = 2) -> Optional[str]:
    node = _node_bin()
    runner = _runner_path()
    if not node or not runner:
        return None
    payload = {'source': source or '', 'actions': actions or []}
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tmp:
            tmp.write(json.dumps(payload))
            tmp.flush()
            tmp_path = tmp.name
        try:
            # Use subprocess directly to pass a strict timeout and avoid env pollution
            proc = subprocess.run([node, str(runner), tmp_path], cwd=str(runner.parent), timeout=max(1, int(timeout_s)), text=True, capture_output=True, check=False)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        out = proc.stdout.strip() if proc.stdout else ''
        if not out:
            return None
        data = json.loads(out)
        if isinstance(data, dict):
            # Print runner messages (one-line action-scoped) when present
            try:
                for m in (data.get('messages') or []):
                    print(f"[js.ts] {m.get('level','info')} action={m.get('action')} index={m.get('index')} file={m.get('file')} reason={m.get('reason')}")
            except Exception:
                pass
            if data.get('ok') and isinstance(data.get('source'), str):
                return data['source']
            if not data.get('ok'):
                # Bridge error: print concise one-liner
                msg = data.get('error') or 'runner-error'
                ctx = []
                if data.get('action'):
                    ctx.append(f"action={data.get('action')}")
                if data.get('index') is not None:
                    ctx.append(f"index={data.get('index')}")
                if data.get('file'):
                    ctx.append(f"file={data.get('file')}")
                print(f"[js.ts] error {msg} {' '.join(ctx)}")
                return None
    except Exception:
        return None
    return None


def apply_actions_js_ts_node_multi(files: Dict[str, str], actions: List[Dict[str, Any]], primary: Optional[str] = None, *, timeout_s: int = 3) -> Optional[Dict[str, str]]:
    """Apply actions across multiple JS/TS files using Node bridge.

    Returns a mapping path->new_source on success, or None if node/runner not available.
    """
    node = _node_bin()
    runner = _runner_path()
    if not node or not runner:
        return None
    arr = [{ 'path': p, 'source': s } for p, s in (files or {}).items()]
    payload = { 'files': arr, 'actions': actions or [], 'primary': primary or (arr[0]['path'] if arr else 'file.tsx') }
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tmp:
            tmp.write(json.dumps(payload))
            tmp.flush()
            tmp_path = tmp.name
        try:
            proc = subprocess.run([node, str(runner), tmp_path], cwd=str(runner.parent), timeout=max(1, int(timeout_s)), text=True, capture_output=True, check=False)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        out = proc.stdout.strip() if proc.stdout else ''
        if not out:
            return None
        data = json.loads(out)
        if isinstance(data, dict):
            try:
                for m in (data.get('messages') or []):
                    print(f"[js.ts] {m.get('level','info')} action={m.get('action')} index={m.get('index')} file={m.get('file')} reason={m.get('reason')}")
            except Exception:
                pass
            if data.get('ok') and isinstance(data.get('files'), list):
                result: Dict[str, str] = {}
                for it in data['files']:
                    try:
                        result[str(it.get('path'))] = str(it.get('source') or '')
                    except Exception:
                        continue
                return result
            if not data.get('ok'):
                msg = data.get('error') or 'runner-error'
                ctx = []
                if data.get('action'):
                    ctx.append(f"action={data.get('action')}")
                if data.get('index') is not None:
                    ctx.append(f"index={data.get('index')}")
                if data.get('file'):
                    ctx.append(f"file={data.get('file')}")
                print(f"[js.ts] error {msg} {' '.join(ctx)}")
                return None
    except Exception:
        return None
    return None
