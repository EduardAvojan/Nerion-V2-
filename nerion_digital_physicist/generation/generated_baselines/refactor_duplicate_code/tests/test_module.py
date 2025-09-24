import pytest
from module import base_computation, select_processor


def test_base_computation():
    assert base_computation(3) == 10


def test_select_processors_equivalent():
    outputs = []
    for idx in range(3 * 2):
        outputs.append(select_processor(idx, 4))
    assert len(set(outputs)) == 1

def test_payload_shape():
    payload = select_processor(0, 2)
    assert payload
