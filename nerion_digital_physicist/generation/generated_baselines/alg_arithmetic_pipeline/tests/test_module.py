import pytest
from module import process_numbers, demo_input


def test_pipeline_runs():
    numbers = demo_input()
    output = process_numbers(numbers)
    assert len(output) == len(numbers)


def test_pipeline_expected_values():
    numbers = [0, 1, 2]
    result = process_numbers(numbers)
    expected = []
    for item in numbers:
        current = float(item)
        current += 3.0
        current *= 2.0
        current = max(0.0, min(100.0, current))
        expected.append(current)
    assert result == expected
