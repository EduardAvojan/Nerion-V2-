from __future__ import annotations

from pathlib import Path
import json
import zipfile

from selfcoder import cli


def _w(p: Path, s: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding='utf-8')
    return p


def test_package_pack_run_builds_zip(tmp_path: Path, monkeypatch):
    # Create minimal artifacts in the current repo (under out/ and .nerion/)
    _w(Path('.nerion/plan_cache.json'), json.dumps({}))
    _w(Path('.nerion/artifacts/plan_000.json'), json.dumps({'ok': True}))
    _w(Path('out/index/index.json'), json.dumps({'files': 1}))
    _w(Path('out/experience/log.jsonl'), '')
    _w(Path('out/learning/prefs.json'), json.dumps({'tool_success_rate': {}}))
    _w(Path('out/voice/latency.jsonl'), json.dumps({'duration_ms': 10}) + '\n')
    _w(Path('out/bench/demo/triage.json'), json.dumps({'failed': []}))
    _w(Path('app/settings.yaml'), 'voice:\n  always_speak: true\n')

    rc = cli.main(['package', 'pack', 'run'])
    assert rc == 0

    # Find the latest bundle under out/package
    pkg_dir = Path('out/package')
    zips = sorted(pkg_dir.glob('run_*.zip'))
    assert zips, 'no run bundle produced'
    bundle = zips[-1]

    # Inspect contents
    with zipfile.ZipFile(bundle, 'r') as z:
        names = set(z.namelist())
        assert '.nerion/plan_cache.json' in names
        assert '.nerion/artifacts/plan_000.json' in names
        assert 'out/index/index.json' in names
        assert 'app/settings.yaml' in names

