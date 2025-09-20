from __future__ import annotations

from pathlib import Path
from selfcoder.analysis import index_api
from selfcoder.analysis import symbols_graph as sgraph


def test_from_import_and_relative_resolution(tmp_path: Path):
    # Create a small package with relative import and from-import
    pkg = tmp_path / 'pkg'
    sub = pkg / 'sub'
    sub.mkdir(parents=True)
    (pkg / '__init__.py').write_text('', encoding='utf-8')
    (sub / '__init__.py').write_text('', encoding='utf-8')
    (sub / 'a.py').write_text('def foo():\n    return 1\n', encoding='utf-8')
    # b.py imports from sub.a using from-import and relative import
    (pkg / 'b.py').write_text('from pkg.sub.a import foo\n\nval = foo()\n', encoding='utf-8')
    # c.py uses relative import (from .sub import a)
    (pkg / 'c.py').write_text('from .sub import a\n\nres = a.foo()\n', encoding='utf-8')
    idx = index_api.build(tmp_path)
    aff = index_api.affected(tmp_path, 'foo', transitive=True)
    assert any(p.endswith('pkg/b.py') for p in aff) or any(p.endswith('pkg/c.py') for p in aff)


def test_multi_hop_depth(tmp_path: Path):
    # a -> b (imports a), c -> b (imports b), expect a affects c with depth >= 2
    (tmp_path / 'a.py').write_text('def foo():\n    return 1\n', encoding='utf-8')
    (tmp_path / 'b.py').write_text('from a import foo\n', encoding='utf-8')
    (tmp_path / 'c.py').write_text('import b\n', encoding='utf-8')
    index_api.build(tmp_path)
    aff1 = sgraph.affected_files_for_symbol('foo', tmp_path, transitive=True, depth=1)
    aff2 = sgraph.affected_files_for_symbol('foo', tmp_path, transitive=True, depth=2)
    assert any(p.endswith('b.py') for p in aff1)
    assert any(p.endswith('c.py') for p in aff2)


def test_method_defs_and_uses(tmp_path: Path):
    # Class with method; separate file calls obj.method(), we expect Class.method to appear in uses when methods enabled
    (tmp_path / 'a.py').write_text('class C:\n    def foo(self):\n        return 1\n', encoding='utf-8')
    (tmp_path / 'b.py').write_text('from a import C\n\nobj = C()\nobj.foo()\n', encoding='utf-8')
    from selfcoder.analysis import symbols as syms
    idx = syms.build_defs_uses(tmp_path, use_cache=False, include_methods=True)
    # Defs should include Class.method
    assert 'C.foo' in idx.get('defs', {})
    # Uses for C.foo should include b.py
    uses = idx.get('uses', {}).get('C.foo', [])
    assert any(str(r.get('file','')).endswith('b.py') for r in uses)
