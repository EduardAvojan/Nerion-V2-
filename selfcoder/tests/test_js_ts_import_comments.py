from __future__ import annotations

import shutil
from pathlib import Path
from selfcoder.actions.js_ts_node import apply_actions_js_ts_node


def test_preserve_comments_in_named_imports(tmp_path: Path):
    if shutil.which('node') is None:
        return
    f = tmp_path / 'a.ts'
    f.write_text("import React, { useState /* keep */ } from 'react';\n", encoding='utf-8')
    actions = [{ 'kind': 'update_import', 'payload': { 'module': 'react', 'named': ['useEffect'] } }]
    out = apply_actions_js_ts_node(f.read_text(encoding='utf-8'), actions)
    if out is not None:
        assert '/* keep */' in out
        assert 'useEffect' in out

