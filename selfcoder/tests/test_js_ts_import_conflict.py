from __future__ import annotations

import shutil
from selfcoder.actions.js_ts_node import apply_actions_js_ts_node


def test_default_conflict_returns_none_when_node_present():
    # Only assert behavior when Node is present (runner returns None on error)
    if shutil.which('node') is None:
        return
    src = """
import React from 'react';
""".lstrip()
    actions = [
        {"kind": "update_import", "payload": {"module": "react", "default": "Foo"}},
    ]
    out = apply_actions_js_ts_node(src, actions)
    # Conflict should cause runner to fail -> wrapper returns None
    assert out is None

