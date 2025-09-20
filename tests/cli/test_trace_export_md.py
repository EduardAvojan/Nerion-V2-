from pathlib import Path
from selfcoder.cli_ext import trace as mod


class Args:
    def __init__(self, since='24h', out='out/report.md'):
        self.since = since
        self.out = out


def test_trace_export_writes_markdown(tmp_path, monkeypatch):
    # Create a tiny experience log
    out_dir = Path('out/experience')
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / 'log.jsonl'
    p.write_text('{"ts": 9999999999, "action":"x","ms":10,"result":"OK"}\n', encoding='utf-8')
    out = tmp_path / 'digest.md'
    rc = mod.cmd_export(Args('1h', str(out)))
    assert rc == 0 and out.exists()
    txt = out.read_text(encoding='utf-8')
    assert '# Nerion Trace Digest' in txt and 'Actions:' in txt
