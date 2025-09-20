from __future__ import annotations

from pathlib import Path

from selfcoder import cli


def _w(p: Path, s: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding='utf-8')
    return p


def test_bench_candidate_selection_prefers_green(tmp_path, monkeypatch):
    task = tmp_path / 'bench_cands'
    _w(task / 'm.py', 'def add(a,b):\n    assert a + b == 3\n    return a+b\n')
    _w(task / 'tests' / 'test_add.py', 'from m import add\n\n\ndef test_add():\n    add(1,1)\n')
    # Plugin with propose_diff_multi: first bad, then good
    plugin_path = Path('plugins/repair_diff.py')
    plugin_path.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "from difflib import unified_diff\n"
        "from pathlib import Path\n"
        "def propose_diff_multi(ctx):\n"
        "    files = ctx.get('files') or []\n"
        "    t = Path(files[0]['path']) if files else None\n"
        "    if not t: return []\n"
        "    txt = t.read_text(encoding='utf-8')\n"
        "    bad = txt.replace('== 3','== 4')\n"
        "    def _disable_asserts(s: str) -> str:\n"
        "        out = []\n"
        "        for ln in s.splitlines():\n"
        "            if ln.lstrip().startswith('assert'):\n"
        "                lead = ln[:len(ln)-len(ln.lstrip())]\n"
        "                out.append(lead + 'pass')\n"
        "            else:\n"
        "                out.append(ln)\n"
        "        return '\\n'.join(out) + '\\n'\n"
        "    good = _disable_asserts(txt)\n"
        "    name = t.name\n"
        "    d1 = ''.join(unified_diff(txt.splitlines(True), good.splitlines(True), fromfile=f'a/{name}', tofile=f'b/{name}'))\n"
        "    d2 = ''.join(unified_diff(txt.splitlines(True), bad.splitlines(True), fromfile=f'a/{name}', tofile=f'b/{name}'))\n"
        "    return [d1, d2]\n"
    )
    plugin_path.write_text(code, encoding='utf-8')
    try:
        monkeypatch.setenv('NERION_BENCH_USE_LIBPYTEST', '1')
        rc = cli.main(['bench', 'repair', '--task', str(task), '--max-iters', '1'])
        assert rc == 0
    finally:
        try:
            plugin_path.unlink()
        except Exception:
            pass
