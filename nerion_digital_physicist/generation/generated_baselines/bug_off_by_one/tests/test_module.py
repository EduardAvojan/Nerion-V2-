import pytest
from module import collect_values


def test_collect_values_expected_length():
    data = list(range(6))
    result = collect_values(data)
    assert len(result) == len(data)


def test_collect_values_matches_input():
    data = list(range(6))
    result = collect_values(data)
    assert result == data
