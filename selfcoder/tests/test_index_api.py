from __future__ import annotations

from pathlib import Path
from selfcoder.analysis import index_api


def test_build_and_query_index(tmp_path: Path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'a.py').write_text('def foo():\n    return 1\n', encoding='utf-8')
    (tmp_path / 'pkg' / 'b.py').write_text('from pkg.a import foo\n\nval = foo()\n', encoding='utf-8')
    idx = index_api.build(tmp_path)
    assert 'defs' in idx and 'uses' in idx and 'imports' in idx
    # Query affected files for symbol foo
    aff = index_api.affected(tmp_path, 'foo', transitive=True)
    # Should include b.py since it imports a.py and calls foo
    assert any(p.endswith('pkg/b.py') for p in aff)

