from __future__ import annotations

from selfcoder.actions.js_ts import apply_actions_js_ts


def test_js_insert_import_named_and_default():
    src = """
// file banner
console.log('hi');
""".lstrip()
    actions = [
        {"kind": "insert_import", "payload": {"module": "react", "default": "React", "named": ["useState"]}},
    ]
    out = apply_actions_js_ts(src, actions)
    # import should appear above code
    assert "import React, { useState } from 'react';" in out.splitlines()[1]


def test_js_insert_import_namespace():
    src = "console.log('x');\n"
    actions = [{"kind": "insert_import", "payload": {"module": "fs", "namespace": "fs"}}]
    out = apply_actions_js_ts(src, actions)
    assert out.startswith("import * as fs from 'fs';\n")


def test_js_insert_class_and_function_and_doc():
    src = "export function f() {}\n"
    actions = [
        {"kind": "insert_function", "payload": {"name": "g", "doc": "G function"}},
        {"kind": "insert_class", "payload": {"name": "Foo", "doc": "Class Foo"}},
    ]
    out = apply_actions_js_ts(src, actions)
    assert "export function g()" in out
    assert "/** Class Foo */" in out
    assert "export class Foo" in out


def test_js_rename_symbol_word_boundary_only():
    src = "const foo = 1; const foobar = 2;\n"
    actions = [{"kind": "rename_symbol", "payload": {"from": "foo", "to": "bar"}}]
    out = apply_actions_js_ts(src, actions)
    # 'foo' replaced but 'foobar' unchanged
    assert "const bar = 1; const foobar = 2;\n" == out

