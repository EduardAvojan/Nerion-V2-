from core.ui.progress import progress, cancelled

def test_progress_context_smoke(capsys):
    with progress("gate"):
        # trivial path; not actually cancelling in unit test
        assert cancelled() is False
    out = capsys.readouterr().err
    assert "gate …" in out and "gate →" in out
