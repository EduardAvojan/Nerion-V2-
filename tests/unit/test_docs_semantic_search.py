from __future__ import annotations

import json
from pathlib import Path

from selfcoder.analysis.knowledge import index as kb


def test_semantic_search_prefers_relevant_chunks(tmp_path, monkeypatch):
    # Prepare chunks directory
    chunks_dir = Path('out/knowledge/chunks')
    chunks_dir.mkdir(parents=True, exist_ok=True)

    a = {
        'topic': 'search:laptops',
        'domain': 'web_search',
        'url': 'https://example.com/a',
        'extract': 'Battery life is 12 hours with the new model.',
        'date': 1,
    }
    b = {
        'topic': 'search:chairs',
        'domain': 'web_search',
        'url': 'https://example.com/b',
        'extract': 'Office chair comfort rating is 5 out of 10.',
        'date': 2,
    }
    (chunks_dir / 'a.json').write_text(json.dumps(a), encoding='utf-8')
    (chunks_dir / 'b.json').write_text(json.dumps(b), encoding='utf-8')

    res = kb.semantic_search('battery life', limit=1)
    assert res and 'Battery' in (res[0].get('extract') or 'Battery')

