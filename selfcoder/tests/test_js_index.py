from __future__ import annotations

from pathlib import Path
from selfcoder.analysis import js_index as JI


def test_js_index_build_and_affected(tmp_path: Path):
    # a.ts exports foo; b.ts imports from './a' and uses foo; c.ts unrelated
    a = tmp_path / 'a.ts'
    b = tmp_path / 'b.ts'
    c = tmp_path / 'c.ts'
    a.write_text("export function foo(){}\n", encoding='utf-8')
    b.write_text("import { foo } from './a';\nconst x = foo();\n", encoding='utf-8')
    c.write_text("export const bar=1;\n", encoding='utf-8')

    idx = JI.build_and_save(tmp_path)
    assert 'foo' in (idx.get('defs') or {})
    aff = JI.affected_files_for_symbol('foo', tmp_path, depth=1)
    # Should include a.ts and b.ts
    assert any(str(a) in s for s in aff)
    assert any(str(b) in s for s in aff)

