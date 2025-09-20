from core.ui.diff import render_unified
def test_render_unified():
    s = render_unified([(' ','a'),('+','b'),('-','c')])
    assert "+ b" in s and "- c" in s and "  a" in s
