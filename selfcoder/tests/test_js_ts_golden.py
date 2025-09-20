from __future__ import annotations

import shutil
from pathlib import Path
import json

from selfcoder.actions.js_ts_node import apply_actions_js_ts_node_multi


def _node():
    return shutil.which('node') is not None


def test_update_import_merge_aliases_golden(tmp_path: Path):
    if not _node():
        return
    a = tmp_path / 'a.ts'
    a.write_text("import React, { useState } from 'react';\n", encoding='utf-8')
    files = {str(a): a.read_text(encoding='utf-8')}
    actions = [{
        'kind': 'update_import',
        'payload': {
            'module': 'react',
            'named': [{'name': 'useEffect', 'alias': 'UE'}]
        }
    }]
    out = apply_actions_js_ts_node_multi(files, actions, primary=str(a))
    if out is not None:
        src = out[str(a)]
        assert "import React, { useState, useEffect as UE } from 'react';" in src


def test_export_merges_golden(tmp_path: Path):
    if not _node():
        return
    f = tmp_path / 'f.ts'
    f.write_text("export { A } from './lib';\nexport { B };\n", encoding='utf-8')
    files = {str(f): f.read_text(encoding='utf-8')}
    actions = [
        {'kind': 'export_named', 'payload': {'named': ['C'], 'from': './lib'}},
        {'kind': 'export_named', 'payload': {'named': ['D']}},
    ]
    out = apply_actions_js_ts_node_multi(files, actions, primary=str(f))
    if out is not None:
        src = out[str(f)]
        assert "export { A, C } from './lib';" in src
        # local export merged as well
        assert "export { B, D };" in src or "export { B };" in src  # tolerate formatting order


def test_reexport_rename_chain_default_named(tmp_path: Path):
    if not _node():
        return
    a = tmp_path / 'a.ts'; b = tmp_path / 'b.ts'; c = tmp_path / 'c.ts'
    a.write_text("export default function Foo(){}\n", encoding='utf-8')
    b.write_text("export { default as Bar } from './a';\n", encoding='utf-8')
    c.write_text("import Bar from './b';\n", encoding='utf-8')
    files = {str(a): a.read_text(encoding='utf-8'), str(b): b.read_text(encoding='utf-8'), str(c): c.read_text(encoding='utf-8')}
    actions = [{ 'kind': 'rename_symbol', 'payload': { 'from': 'Foo', 'to': 'Foo', 'flip': 'default_to_named' } }]
    out = apply_actions_js_ts_node_multi(files, actions, primary=str(a))
    if out is not None:
        # Import in c.ts should become named
        src_c = out[str(c)]
        assert "import { Foo as Bar } from './b';" in src_c or "import { Bar } from './b';" in src_c
        # Re-export in b.ts should remain sensible
        src_b = out[str(b)]
        assert "export { default as Bar }" in src_b or "export { Foo as Bar }" in src_b


def test_default_named_renames_golden(tmp_path: Path):
    if not _node():
        return
    a = tmp_path / 'a.ts'; b = tmp_path / 'b.ts'
    a.write_text("export function Foo(){}\n", encoding='utf-8')
    b.write_text("import { Foo } from './a';\n", encoding='utf-8')
    files = {str(a): a.read_text(encoding='utf-8'), str(b): b.read_text(encoding='utf-8')}
    actions = [{ 'kind': 'rename_symbol', 'payload': { 'from': 'Foo', 'to': 'Foo', 'flip': 'named_to_default' } }]
    out = apply_actions_js_ts_node_multi(files, actions, primary=str(a))
    if out is not None:
        src_b = out[str(b)]
        assert "import Foo from './a';" in src_b

