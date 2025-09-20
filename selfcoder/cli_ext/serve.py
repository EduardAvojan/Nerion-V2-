from __future__ import annotations

import argparse
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any
from pathlib import Path
from core.http.schemas import ok as _ok, err as _err


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # type: ignore[override]
        if self.path == '/version':
            try:
                from app.version import BUILD_TAG
            except Exception:
                BUILD_TAG = 'unknown'
            self._send(200, _ok({'version': BUILD_TAG}))
            return
        self._send(404, _err('NOT_FOUND', 'not found', self.path))

    def do_POST(self):  # type: ignore[override]
        length = int(self.headers.get('Content-Length') or 0)
        data = self.rfile.read(length) if length > 0 else b'{}'
        try:
            payload = json.loads(data.decode('utf-8'))
        except Exception:
            payload = {}
        if self.path == '/patch/preview':
            plan = payload if isinstance(payload, dict) else {}
            try:
                from selfcoder.orchestrator import _apply_actions_preview as preview, _unified_diff_for_file as _ud
                actions = plan.get('actions') or []
                files = []
                if plan.get('files'):
                    files = [Path(p) for p in plan['files']]
                elif plan.get('target_file'):
                    files = [Path(plan['target_file'])]
                pv = preview(files, actions)
                diffs = {str(p): _ud(p, old, new) for p, (old, new) in pv.items()}
                self._send(200, _ok({'diffs': diffs}))
                return
            except Exception as e:
                self._send(500, _err('SERVER_ERROR', str(e), self.path))
                return
        if self.path == '/patch/apply':
            plan = payload if isinstance(payload, dict) else {}
            try:
                from selfcoder.orchestrator import run_actions_on_files as apply_files
                actions = plan.get('actions') or []
                targets = []
                if plan.get('files'):
                    targets = [Path(p) for p in plan['files']]
                elif plan.get('target_file'):
                    targets = [Path(plan['target_file'])]
                changed = apply_files(targets, actions, dry_run=False)
                self._send(200, _ok({'applied': [str(p) for p in changed]}))
                return
            except Exception as e:
                self._send(500, _err('SERVER_ERROR', str(e), self.path))
                return
        if self.path == '/review':
            plan = payload if isinstance(payload, dict) else {}
            try:
                from selfcoder.orchestrator import _apply_actions_preview as preview
                from selfcoder.reviewers.reviewer import review_predicted_changes
                from ops.security import fs_guard as _fg
                actions = plan.get('actions') or []
                targets = []
                if plan.get('files'):
                    targets = [Path(p) for p in plan['files']]
                elif plan.get('target_file'):
                    targets = [Path(plan['target_file'])]
                if not actions or not targets:
                    self._send(400, _err('BAD_REQUEST', 'Missing actions or target_file/files', self.path))
                    return
                previews = preview(targets, actions)
                predicted = {p.as_posix(): new for p, (_old, new) in previews.items()}
                rep = review_predicted_changes(predicted, _fg.infer_repo_root(Path('.')))
                self._send(200, _ok({'review': rep}))
                return
            except Exception as e:
                self._send(500, _err('SERVER_ERROR', str(e), self.path))
                return
        self._send(404, _err('NOT_FOUND', 'not found', self.path))


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('serve', help='start a simple local HTTP server for IDE integration')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    def _run(args: argparse.Namespace) -> int:
        addr = (getattr(args, 'host', '127.0.0.1'), int(getattr(args, 'port', 8765)))
        srv = ThreadingHTTPServer(addr, _Handler)
        print(f"[serve] listening on http://{addr[0]}:{addr[1]}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    p.set_defaults(func=_run)
