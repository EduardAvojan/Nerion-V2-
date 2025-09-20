from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.chat.memory_bridge import LongTermMemory
from app.chat.memory_session import SessionCache


def _fresh_repo_path(relative: str) -> Path:
    path = Path('tmp/tests') / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return path


def test_memory_namespace_backoff():
    mem_path = _fresh_repo_path('memory_ns.json')

    primary = LongTermMemory(path=str(mem_path), ns={'user': 'alice', 'workspace': 'ws', 'project': 'proj'})
    primary.add_fact('Alice prefers tea.', scope='long', confidence=0.9)

    workspace = LongTermMemory(path=str(mem_path), ns={'user': 'alice', 'workspace': 'ws', 'project': 'proj-other'})
    workspace.add_fact('Workspace shared preference.', scope='long', confidence=0.8)

    global_mem = LongTermMemory(path=str(mem_path), ns={'user': 'global', 'workspace': 'default', 'project': 'default'})
    global_mem.add_fact('Global hint for everyone.', scope='long', confidence=0.7)

    reader = LongTermMemory(path=str(mem_path), ns={'user': 'alice', 'workspace': 'ws', 'project': 'proj'})
    results = reader.find_relevant('prefers tea', k=5)

    assert results
    assert results[0]['fact'] == 'Alice prefers tea.'


def test_memory_quarantine(monkeypatch, tmp_path):
    q_path = Path('out/memory/quarantine.jsonl')
    original = None
    if q_path.exists():
        original = q_path.read_text(encoding='utf-8')

    monkeypatch.setenv('NERION_MEMORY_STRICT_PIIGATE', '1')
    monkeypatch.setenv('NERION_MEMORY_QUARANTINE', '1')

    mem_path = _fresh_repo_path('memory_quarantine.json')
    memory = LongTermMemory(path=str(mem_path))
    added = memory.add_fact('SSN 123-45-6789 belongs to someone.', confidence=0.9)

    assert added is False
    assert not any(not m.get('deleted') for m in memory.memories)
    assert q_path.exists()

    payload = q_path.read_text(encoding='utf-8').strip().splitlines()
    assert payload
    record = json.loads(payload[-1])
    assert record['reasons'] and 'item' in record

    if original is None:
        q_path.unlink(missing_ok=True)
    else:
        q_path.write_text(original, encoding='utf-8')


def test_prune_utility_evicts_low_score(monkeypatch, tmp_path):
    monkeypatch.setenv('NERION_MEMORY_TIER_HOT', '1')
    monkeypatch.setenv('NERION_MEMORY_TIER_WARM', '0')
    monkeypatch.setenv('NERION_MEMORY_TIER_COLD', '0')

    mem_path = _fresh_repo_path('memory_prune.json')
    memory = LongTermMemory(path=str(mem_path))
    memory.add_fact('Keep me', score=5.0, scope='short')
    memory.add_fact('Drop me', score=0.1, scope='short')

    summary = memory.prune()
    assert summary['removed'] >= 1

    survivors = [m for m in memory.memories if not m.get('deleted')]
    assert len(survivors) == 1
    assert survivors[0]['fact'] == 'Keep me'


def test_consolidate_creates_canonical_entry(monkeypatch, tmp_path):
    monkeypatch.setenv('NERION_MEMORY_CONSOLIDATE', '1')
    monkeypatch.setenv('NERION_MEMORY_CLUSTER_MIN', '3')

    mem_path = _fresh_repo_path('memory_consolidate.json')
    memory = LongTermMemory(path=str(mem_path))

    monkeypatch.setattr(LongTermMemory, "_source_hash", lambda self, text: "bucket")

    memory.add_fact('User likes jazz.', scope='short')
    memory.add_fact('User likes jazz concerts.', scope='short')
    memory.add_fact('User likes jazz festivals.', scope='short')

    result = memory.consolidate()
    assert result['created'] == 1

    canonical = [m for m in memory.memories if m.get('provenance') == 'consolidate']
    assert canonical and canonical[0]['scope'] == 'long'

    superseded = [m for m in memory.memories if m.get('superseded_by')]
    assert len(superseded) >= 2


def test_session_cache_persistence():
    session_path = _fresh_repo_path('session.json')
    ns = {'user': 'tester', 'workspace': 'ws', 'project': 'proj'}

    cache = SessionCache(path=str(session_path), ns=ns, max_turns=4)
    cache.record_turn('user', 'Hello there')
    cache.upsert_short_fact('User enjoys coffee.', tags=['food'], score=1.0, ttl_days=10)
    cache.save()
    assert session_path.exists()

    cache2 = SessionCache(path=str(session_path), ns=ns, max_turns=4)
    cache2.load()
    assert cache2.state['turns'] and cache2.state['short_facts']

    promos = cache2.decay_and_prune(decay_per_day=0.0, default_ttl_days=30, promotion_threshold=0.5)
    assert promos and promos[0]['fact'] == 'User enjoys coffee.'
