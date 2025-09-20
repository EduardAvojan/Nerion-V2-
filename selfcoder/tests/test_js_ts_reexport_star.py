from __future__ import annotations

import shutil
from pathlib import Path
from selfcoder.actions.js_ts_node import apply_actions_js_ts_node_multi


def test_reexport_star_chain_importer_updates(tmp_path: Path):
    if shutil.which('node') is None:
        return
    a = tmp_path / 'a.ts'; b = tmp_path / 'b.ts'; c = tmp_path / 'c.ts'
    a.write_text("export function Foo(){}\n", encoding='utf-8')
    b.write_text("export * from './a';\n", encoding='utf-8')
    c.write_text("import { Foo } from './b';\nconst x = Foo();\n", encoding='utf-8')
    files = {str(a): a.read_text(encoding='utf-8'), str(b): b.read_text(encoding='utf-8'), str(c): c.read_text(encoding='utf-8')}
    # Flip named_to_default at the source; importer should become default import
    actions = [{ 'kind': 'rename_symbol', 'payload': { 'from': 'Foo', 'to': 'Foo', 'flip': 'named_to_default' } }]
    out = apply_actions_js_ts_node_multi(files, actions, primary=str(a))
    if out is not None:
        src_c = out[str(c)]
        assert "import Foo from './b';" in src_c

