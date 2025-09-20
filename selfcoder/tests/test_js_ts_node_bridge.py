from __future__ import annotations

import shutil
from selfcoder.actions.js_ts_node import apply_actions_js_ts_node


def test_node_bridge_graceful_fallback_when_unavailable():
    # When node or runner is unavailable, wrapper returns None
    src = "console.log('x');\n"
    actions = [{"kind": "insert_function", "payload": {"name": "f"}}]
    # Temporarily simulate missing node by checking actual environment
    node = shutil.which('node')
    out = apply_actions_js_ts_node(src, actions)
    if node is None:
        assert out is None
    else:
        # If node exists, we still may not have ts-morph; accept None as fallback
        assert (out is None) or isinstance(out, str)

