from pathlib import Path

from selfcoder.cli_ext import scan as mod
from selfcoder.security import Finding


class Args:
    def __init__(self, files, json=False, outdir='backups/security', fail_on='none'):
        self.files = files
        self.json = json
        self.outdir = outdir
        self.fail_on = fail_on


def test_scan_prints_unified_summary(tmp_path, monkeypatch, capsys):
    # Create a dummy file
    f = tmp_path / 'x.py'
    f.write_text('print(1)\n', encoding='utf-8')

    # Monkeypatch scanner to return one high severity finding
    def fake_scan_source(src, filename, repo_root):
        return [Finding(rule_id='X', severity='high', message='m', filename=str(filename), line=1, evidence='')]
    monkeypatch.setattr('selfcoder.security.scanner.scan_source', fake_scan_source, raising=False)

    rc = mod.cmd_scan(Args([str(f)]))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'scan: summary' in out and 'findings=1' in out and 'worst=high' in out

