from __future__ import annotations

from typing import Optional
import shutil
from selfcoder.actions.js_ts_node import apply_actions_js_ts_node


def _maybe_assert_contains(out: Optional[str], needle: str):
    if out is not None:
        assert needle in out


def test_node_update_import_export_and_types():
    # Skip assertions if node/ts-morph not available (out=None)
    src = """
// base
const x = 1;
""".lstrip()
    actions = [
        {"kind": "update_import", "payload": {"module": "react", "default": "React", "named": [{"name": "useState", "alias": "useS"}], "isType": True}},
        {"kind": "insert_interface", "payload": {"name": "Foo", "doc": "IF"}},
        {"kind": "insert_type_alias", "payload": {"name": "Alias", "type": "string"}},
        {"kind": "export_named", "payload": {"named": ["x"]}},
        {"kind": "export_default", "payload": {"name": "x"}},
    ]
    out = apply_actions_js_ts_node(src, actions)
    # If node present, verify expected constructs
    _maybe_assert_contains(out, "import type")
    _maybe_assert_contains(out, "export interface Foo")
    _maybe_assert_contains(out, "export type Alias = string")
    _maybe_assert_contains(out, "export { x }")
    _maybe_assert_contains(out, "export default x")


def test_node_scoped_rename_symbol():
    src = """
export function foo() { return 1; }
function main() { return foo(); }
""".lstrip()
    actions = [{"kind": "rename_symbol", "payload": {"from": "foo", "to": "bar"}}]
    out = apply_actions_js_ts_node(src, actions)
    if out is not None:
        assert "function bar()" in out
        assert "bar();" in out


def test_flip_default_to_named_updates_importers(tmp_path):
    # Only run when node present
    import shutil as _sh
    if _sh.which('node') is None:
        return
    files = {
        str(tmp_path / 'a.ts'): "export default function Foo(){}\n",
        str(tmp_path / 'b.ts'): "import Foo from './a';\nconst z = Foo();\n",
    }
    actions = [{"kind": "rename_symbol", "payload": {"from": "Foo", "to": "Foo", "flip": "default_to_named"}}]
    from selfcoder.actions.js_ts_node import apply_actions_js_ts_node_multi
    out = apply_actions_js_ts_node_multi(files, actions, primary=str(tmp_path / 'a.ts'))
    if out is not None:
        b = out[str(tmp_path / 'b.ts')]
        assert "import { Foo } from './a';" in b


def test_flip_named_to_default_updates_importers(tmp_path):
    import shutil as _sh
    if _sh.which('node') is None:
        return
    files = {
        str(tmp_path / 'a.ts'): "export function Foo(){}\n",
        str(tmp_path / 'b.ts'): "import { Foo } from './a';\nconst z = Foo();\n",
    }
    actions = [{"kind": "rename_symbol", "payload": {"from": "Foo", "to": "Foo", "flip": "named_to_default"}}]
    from selfcoder.actions.js_ts_node import apply_actions_js_ts_node_multi
    out = apply_actions_js_ts_node_multi(files, actions, primary=str(tmp_path / 'a.ts'))
    if out is not None:
        b = out[str(tmp_path / 'b.ts')]
        assert "import Foo from './a';" in b


def test_touch_jsx_props_flag(tmp_path):
    import shutil as _sh
    if _sh.which('node') is None:
        return
    files = {
        str(tmp_path / 'c.tsx'): "const el = <Comp foo={1} />;\n",
    }
    actions = [{"kind": "rename_symbol", "payload": {"from": "foo", "to": "bar", "touchJSXProps": True}}]
    from selfcoder.actions.js_ts_node import apply_actions_js_ts_node_multi
    out = apply_actions_js_ts_node_multi(files, actions, primary=str(tmp_path / 'c.tsx'))
    if out is not None:
        c = out[str(tmp_path / 'c.tsx')]
        assert "<Comp bar={1} />" in c
