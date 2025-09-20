from __future__ import annotations

from pathlib import Path

from selfcoder.orchestrator import run_actions_on_file


def _w(p: Path, s: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding='utf-8')
    return p


def test_js_add_doc_and_insert_function(tmp_path: Path):
    f = tmp_path / 'mod.js'
    _w(f, 'function add(a,b){ return a+b; }\n')
    actions = [
        {"kind": "add_module_docstring", "payload": {"doc": "Module JS docs"}},
        {"kind": "insert_function", "payload": {"name": "greet", "doc": "Say hello"}},
    ]
    ok = run_actions_on_file(f, actions, dry_run=False)
    assert ok is True
    out = f.read_text(encoding='utf-8')
    assert out.lstrip().startswith('/** Module JS docs */')
    assert 'export function greet()' in out


def test_ts_rename_symbol(tmp_path: Path):
    f = tmp_path / 'types.ts'
    _w(f, 'export function add(a:number,b:number){ return a+b; }\n')
    actions = [
        {"kind": "rename_symbol", "payload": {"from": "add", "to": "sum"}},
    ]
    ok = run_actions_on_file(f, actions, dry_run=False)
    assert ok is True
    out = f.read_text(encoding='utf-8')
    assert 'export function sum(' in out

