def add(a, b):
    return a + b

def test_add_broken():
    # This test is deliberately broken to demonstrate the autonomous fixer
    assert add(2, 2) == 4