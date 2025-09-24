from pathlib import Path

import torch

from nerion_digital_physicist.agent.data import create_graph_data_object
from nerion_digital_physicist.agent.semantics import DEFAULT_DIMENSION, get_global_embedder


def test_semantic_features_dimension(tmp_path):
    embedder = get_global_embedder()
    repo_root = Path(__file__).resolve().parents[2]
    file_path = repo_root / "nerion_digital_physicist" / "environment" / "logic_v2.py"
    graph = create_graph_data_object(file_path, embedder=embedder)
    assert graph.x.shape[1] == 7 + DEFAULT_DIMENSION
    assert isinstance(graph.x, torch.Tensor)


def test_semantic_cache_reuse(tmp_path):
    embedder = get_global_embedder()
    repo_root = Path(__file__).resolve().parents[2]
    file_path = repo_root / "nerion_digital_physicist" / "environment" / "logic_v2.py"
    graph1 = create_graph_data_object(file_path, embedder=embedder)
    graph2 = create_graph_data_object(file_path, embedder=embedder)
    assert torch.allclose(graph1.x, graph2.x)
