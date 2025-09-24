"""Phase 2 data pipeline: convert code graphs into PyG Data objects."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from .semantics import SemanticEmbedder, get_global_embedder

LOGIC_PATH = Path(__file__).resolve().parent.parent / "environment" / "logic_v2.py"


def get_node_features(node_name: str, graph: nx.DiGraph, code_ast: ast.AST) -> list[float]:
    """Generate a feature vector for a single graph node."""
    features: list[float] = []

    node_type = graph.nodes[node_name].get("node_type", "unknown")
    features.extend(
        [
            1 if node_type == "function" else 0,
            1 if node_type == "variable" else 0,
            1 if node_type == "call" else 0,
        ]
    )

    features.append(len(node_name))

    if node_type == "function":
        line_count = 0
        arg_count = 0
        total_arg_length = 0
        for node in ast.walk(code_ast):
            if isinstance(node, ast.FunctionDef) and node.name == node_name:
                start_line = node.lineno
                end_line = max(
                    (
                        child.lineno
                        for child in ast.walk(node)
                        if hasattr(child, "lineno")
                    ),
                    default=start_line,
                )
                line_count = end_line - start_line + 1

                for arg in node.args.args:
                    arg_count += 1
                    total_arg_length += len(arg.arg)
                break
        avg_arg_length = (total_arg_length / arg_count) if arg_count else 0
        features.extend([float(line_count), float(arg_count), float(avg_arg_length)])
    else:
        features.extend([0.0, 0.0, 0.0])

    return features


def _extract_source_segment(node: ast.AST, source: str) -> str:
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        lines = source.splitlines()
        start = max(node.lineno - 1, 0)
        end = max(getattr(node, "end_lineno", node.lineno) - 1, start)
        return "\n".join(lines[start : end + 1])
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive fallback
        return ""


def _build_graph_from_ast(code_ast: ast.AST) -> tuple[nx.DiGraph, Dict[str, ast.AST]]:
    """Create a directed graph describing the code structure."""

    graph = nx.DiGraph()
    node_lookup: Dict[str, ast.AST] = {}
    for node in ast.walk(code_ast):
        if isinstance(node, ast.FunctionDef):
            graph.add_node(node.name, node_type="function")
            node_lookup[node.name] = node
    return graph, node_lookup


def _featurize_graph(
    graph: nx.DiGraph,
    code_ast: ast.AST,
    node_lookup: Dict[str, ast.AST],
    source_code: str,
    embedder: Optional[SemanticEmbedder] = None,
) -> torch.Tensor:
    """Generate the feature tensor for each node in the graph."""

    node_features: list[list[float]] = []
    embedding_dim = embedder.dimension if embedder else 0

    for node_name in graph.nodes():
        base_features = get_node_features(node_name, graph, code_ast)
        if embedder:
            ast_node = node_lookup.get(node_name)
            snippet = _extract_source_segment(ast_node, source_code) if ast_node else ""
            embedding = embedder.embed(node_name, snippet)
            base_features.extend(embedding)
        node_features.append(base_features)

    if not node_features:
        total_dim = 7 + embedding_dim
        return torch.zeros((0, total_dim), dtype=torch.float)

    return torch.tensor(node_features, dtype=torch.float)


def create_graph_data_from_ast(
    code_ast: ast.AST,
    source_code: str,
    *,
    embedder: Optional[SemanticEmbedder] = None,
) -> Data:
    """Produce a PyG `Data` object directly from a parsed AST."""

    graph, node_lookup = _build_graph_from_ast(code_ast)
    embedder = embedder or get_global_embedder()
    feature_tensor = _featurize_graph(graph, code_ast, node_lookup, source_code, embedder)

    pyg_graph = from_networkx(graph)
    pyg_graph.x = feature_tensor

    return pyg_graph


def create_graph_data_from_source(
    source_code: str,
    *,
    embedder: Optional[SemanticEmbedder] = None,
) -> Data:
    """Produce graph data directly from source text."""

    code_ast = ast.parse(source_code)
    return create_graph_data_from_ast(code_ast, source_code, embedder=embedder)


def create_graph_data_object(
    file_path: str | Path = LOGIC_PATH,
    *,
    embedder: Optional[SemanticEmbedder] = None,
) -> Data:
    """Build a PyTorch Geometric Data object from a file path."""

    file_path = Path(file_path)
    code = file_path.read_text(encoding="utf-8")
    return create_graph_data_from_source(code, embedder=embedder)


def main():
    """Run the data pipeline and display the generated features."""
    graph_data = create_graph_data_object(LOGIC_PATH)

    print("✅ Data Pipeline Ran Successfully!\n")
    print("--- PyTorch Geometric Data Object ---")
    print(graph_data)
    print("\n--- Node Features (data.x) ---")
    print(graph_data.x)
    print("\nShape:", graph_data.x.shape)
    print("-----------------------------------")
    print("\nDescription:")
    print("Each row is a node in our graph (a function).")
    print("Each column is a feature describing that node.")


if __name__ == "__main__":
    main()
