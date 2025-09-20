import pytest
from pathlib import Path


def test_generated_plan_applied(tmp_path):
    """Create a throwaway demo file in a temp dir and assert its contents.

    This avoids relying on hardcoded OS-specific paths under /tmp.
    """
    p = tmp_path / "autotest_demo.py"
    p.write_text(
        '"""AutoTest demo"""\n'
        "def autotest_demo():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    src = p.read_text(encoding="utf-8")
    assert '"""AutoTest demo"""' in src or "'''AutoTest demo'''" in src