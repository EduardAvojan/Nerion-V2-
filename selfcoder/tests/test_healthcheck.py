from selfcoder.healthcheck import run_all

def test_healthcheck_all():
    assert run_all(verbose=True) is True
