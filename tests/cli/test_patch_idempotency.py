import json
from pathlib import Path

from selfcoder.orchestrator import apply_plan


def test_apply_plan_idempotent_ast_action(tmp_path):
    target = tmp_path / 'm.py'
    target.write_text('x=1\n', encoding='utf-8')
    plan = {
        'actions': [
            {'kind': 'add_module_docstring', 'payload': {'doc': 'Test module.'}},
        ],
        'target_file': str(target),
    }
    # First apply should modify the file
    out1 = apply_plan(plan, dry_run=False)
    assert target in out1 and target.read_text(encoding='utf-8').lstrip().startswith('"""')
    # Second apply should be a no-op
    out2 = apply_plan(plan, dry_run=False)
    assert out2 == []
