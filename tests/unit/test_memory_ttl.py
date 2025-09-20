from __future__ import annotations

from app.chat.memory_bridge import LongTermMemory


def test_memory_ttl_prune_expires(tmp_path):
    # Use a path inside repo-relative tmp
    mem_path = 'tmp/test_memory.json'
    m = LongTermMemory(mem_path)
    m.erase_all()
    m.add_fact('User likes espresso.', scope='long')
    assert any('espresso' in it.get('fact','') for it in m.list_memories())
    assert m.set_ttl_for_text('espresso', 0) is True  # expires immediately
    out = m.prune()
    assert any(k in out for k in ('removed','total'))
    assert not any('espresso' in it.get('fact','') for it in m.list_memories())

