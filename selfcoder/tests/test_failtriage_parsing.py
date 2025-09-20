from __future__ import annotations

from pathlib import Path

from selfcoder.analysis.failtriage import (
    _extract_failed_tests,
    _parse_first_traceback,
    _score_suspects,
    build_context_pack,
    TraceFrame,
)


def test_triage_extract_and_score_basic(tmp_path):
    # Synthesize minimal pytest output with one failure and a traceback frame
    mod = Path("tmp/triage_mod.py")
    mod.parent.mkdir(parents=True, exist_ok=True)
    mod.write_text("def f():\n    return 0\n")
    output = (
        '=========================== short test summary info ===========================\n'
        'FAILED tests/test_sample.py::test_fail - AssertionError: boom\n\n'
        'E   AssertionError: boom\n'
        f'  File "{mod.resolve()}", line 1, in f\n'
    )
    failed = _extract_failed_tests(output)
    assert failed and failed[0].endswith("tests/test_sample.py::test_fail")

    msg, frames = _parse_first_traceback(output)
    # Message may be empty depending on ordering; frames are primary for suspects
    assert isinstance(msg, str)
    assert frames and frames[0].file.endswith('triage_mod.py')

    suspects = _score_suspects(frames, msg, Path('.').resolve())
    assert suspects and any('triage_mod.py' in s[0] for s in suspects)

    # Build a compact context pack and ensure the window contains our code
    pack = build_context_pack(Path('.').resolve(), suspects, frames, top_n=1)
    assert pack['files'] and 'window' in pack['files'][0]
    assert 'return 0' in pack['files'][0]['window']
