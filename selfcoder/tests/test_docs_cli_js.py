from __future__ import annotations
import argparse
import json
from urllib.parse import quote

import pytest

# Skip whole module if playwright isn't available
pytest.importorskip("playwright", reason="Playwright not installed; skipping JS rendering tests")

from selfcoder.cli_ext import docs_cli


def _run_cmd(func, ns: argparse.Namespace):
    import io, sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = func(ns)
    finally:
        sys.stdout = old
    out = buf.getvalue()
    return code, json.loads(out)


def _make_data_url(html: str) -> str:
    return "data:text/html," + quote(html)


JS_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <script>
      setTimeout(function(){
        document.body.innerHTML = '<h1>JS Title</h1><p>Rendered content appears</p>';
      }, 100);
    </script>
  </head>
  <body>
    Loading...
  </body>
</html>
"""


def test_docs_read_url_render_renders_js():
    url = _make_data_url(JS_HTML)
    # With render=True, we expect the JS-updated content
    code, data = _run_cmd(
        docs_cli.cmd_read,
                    argparse.Namespace(path=None, url=url, timeout=5, render=True, render_timeout=5, selector=None),    )
    assert code == 0
    text = data.get("text", "")
    # Should include the JS-rendered title/content, not just the initial "Loading..."
    assert "JS Title" in text or "Rendered content appears" in text


def test_docs_read_url_without_render_shows_initial_html():
    url = _make_data_url(JS_HTML)
    # Without render, we fetch the raw HTML and strip — no JS execution
    code, data = _run_cmd(
        docs_cli.cmd_read,
        argparse.Namespace(path=None, url=url, timeout=5, render=False, render_timeout=2, selector=None),
    )
    assert code == 0
    text = data.get("text", "")
    assert "Loading..." in text
