"""Pytest suite for the scope-sensitive logic module."""

import pytest

from logic_v2 import add_scoped, subtract_scoped, value as global_value


def test_add_scoped():
    assert add_scoped(10, 5) == 15


def test_subtract_scoped():
    assert subtract_scoped(10, 5) == 5


def test_global_value():
    assert global_value == 100
