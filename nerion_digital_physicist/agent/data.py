"""Phase 2 data pipeline: convert code graphs into PyG Data objects."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data

from .semantics import SemanticEmbedder, get_global_embedder

LOGIC_PATH = Path(__file__).resolve().parent.parent / "environment" / "logic_v2.py"

# Edge roles captured in the graph; expand as new semantics are modelled.
EDGE_ROLE_TO_INDEX: Dict[str, int] = {
    "sequence": 0,
    "call": 1,
    "shared_symbol": 2,
}


def _function_parameter_names(node: ast.FunctionDef) -> Set[str]:
    """Return all parameter identifiers for ``node``."""

    params: Set[str] = set()

    def _collect(args: Iterable[ast.arg]) -> None:
        for arg in args:
            params.add(arg.arg)

    _collect(node.args.args)
    if node.args.vararg:
        params.add(node.args.vararg.arg)
    _collect(node.args.kwonlyargs)
    if node.args.kwarg:
        params.add(node.args.kwarg.arg)

    return params


def _collect_function_facts(node: ast.FunctionDef, source: str) -> Dict[str, object]:
    """Gather structural statistics and symbol usage for a function definition."""

    params = _function_parameter_names(node)
    docstring = ast.get_docstring(node) or ""
    doc_len = len(docstring.strip())
    body_nodes = list(ast.walk(node))

    branch_nodes = sum(
        isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match, ast.BoolOp))
        for n in body_nodes
    )
    call_nodes: List[Tuple[str, ast.Call]] = []
    return_nodes = sum(isinstance(n, ast.Return) for n in body_nodes)

    reads: Set[str] = set()
    writes: Set[str] = set()

    for n in body_nodes:
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name):
                call_nodes.append((func.id, n))
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Store):
                if n.id not in params:
                    writes.add(n.id)
            elif isinstance(n.ctx, ast.Load):
                if n.id not in params:
                    reads.add(n.id)

    call_targets = [name for name, _ in call_nodes]

    line_count = 0
    arg_count = len(params)
    total_arg_length = sum(len(p) for p in params)
    if hasattr(node, "lineno"):
        end_line = max(
            (
                getattr(child, "end_lineno", getattr(child, "lineno", node.lineno))
                for child in ast.walk(node)
            ),
            default=node.lineno,
        )
        line_count = end_line - node.lineno + 1

    avg_arg_length = (total_arg_length / arg_count) if arg_count else 0.0
    cyclomatic = branch_nodes + 1

    return {
        "lineno": getattr(node, "lineno", 0),
        "order_index": getattr(node, "lineno", 0),  # overwritten later with ordinal position
        "line_count": float(line_count),
        "arg_count": float(arg_count),
        "avg_arg_length": float(avg_arg_length),
        "docstring_length": float(doc_len),
        "branch_count": float(branch_nodes),
        "call_count": float(len(call_nodes)),
        "return_count": float(return_nodes),
        "cyclomatic_complexity": float(cyclomatic),
        "call_targets": call_targets,
        "reads": reads,
        "writes": writes,
    }


def _add_edge_with_role(graph: nx.DiGraph, source: str, target: str, role: str) -> None:
    """Create or augment an edge with a semantic role label."""

    if source == target:
        return
    if graph.has_edge(source, target):
        existing = set(graph[source][target].get("edge_roles", []))
        existing.add(role)
        graph[source][target]["edge_roles"] = sorted(existing)
    else:
        graph.add_edge(source, target, edge_roles=[role])


def get_node_features(node_name: str, graph: nx.DiGraph) -> List[float]:
    """Generate a feature vector for a single graph node."""

    features: List[float] = []

    node_data = graph.nodes[node_name]
    node_type = node_data.get("node_type", "unknown")
    metadata: Dict[str, object] = node_data.get("metadata", {})  # type: ignore[assignment]

    features.extend(
        [
            1.0 if node_type == "function" else 0.0,
            1.0 if node_type == "class" else 0.0,
            1.0 if node_type == "call" else 0.0,
        ]
    )

    features.append(float(len(node_name)))

    if node_type == "function":
        features.extend(
            [
                float(metadata.get("line_count", 0.0)),
                float(metadata.get("arg_count", 0.0)),
                float(metadata.get("avg_arg_length", 0.0)),
                float(metadata.get("docstring_length", 0.0)),
                float(metadata.get("branch_count", 0.0)),
                float(metadata.get("call_count", 0.0)),
                float(metadata.get("return_count", 0.0)),
                float(metadata.get("cyclomatic_complexity", 0.0)),
                float(metadata.get("order_index", 0.0)),
                float(metadata.get("order_normalised", 0.0)),
                float(len(metadata.get("call_targets", []))),
                float(len(metadata.get("reads", []))),
                float(len(metadata.get("writes", []))),
            ]
        )
    else:
        features.extend([0.0] * 13)

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


def _build_graph_from_ast(
    code_ast: ast.AST,
    source_code: str,
) -> Tuple[nx.DiGraph, Dict[str, ast.FunctionDef]]:
    """Create a directed graph describing the code structure."""

    graph = nx.DiGraph()
    node_lookup: Dict[str, ast.FunctionDef] = {}
    metadata: Dict[str, Dict[str, object]] = {}

    for node in ast.walk(code_ast):
        if isinstance(node, ast.FunctionDef):
            node_lookup[node.name] = node
            facts = _collect_function_facts(node, source_code)
            metadata[node.name] = facts
            graph.add_node(node.name, node_type="function", metadata=facts)

    if not metadata:
        return graph, node_lookup

    ordered = sorted(metadata.items(), key=lambda item: item[1].get("lineno", 0))

    denominator = float(max(len(ordered) - 1, 1))

    for ordinal, (name, facts) in enumerate(ordered):
        facts["order_index"] = float(ordinal)
        facts["order_normalised"] = float(ordinal / denominator)
        graph.nodes[name]["metadata"] = facts
        if ordinal > 0:
            prev_name = ordered[ordinal - 1][0]
            _add_edge_with_role(graph, prev_name, name, "sequence")

    defined = set(metadata.keys())

    for name, facts in metadata.items():
        for target in facts.get("call_targets", []):
            if target in defined:
                _add_edge_with_role(graph, name, target, "call")

    symbol_writers: Dict[str, Set[str]] = defaultdict(set)
    symbol_readers: Dict[str, Set[str]] = defaultdict(set)

    for name, facts in metadata.items():
        for sym in facts.get("writes", []):
            symbol_writers[str(sym)].add(name)
        for sym in facts.get("reads", []):
            symbol_readers[str(sym)].add(name)

    for symbol, writers in symbol_writers.items():
        readers = symbol_readers.get(symbol, set())
        if not readers:
            continue
        for writer in writers:
            for reader in readers:
                if writer != reader:
                    _add_edge_with_role(graph, writer, reader, "shared_symbol")

    # Ensure metadata collections are JSON-serialisable downstream.
    for facts in metadata.values():
        facts["call_targets"] = sorted(str(item) for item in facts.get("call_targets", []))
        facts["reads"] = sorted(str(item) for item in facts.get("reads", []))
        facts["writes"] = sorted(str(item) for item in facts.get("writes", []))

    return graph, node_lookup


def _featurize_graph(
    graph: nx.DiGraph,
    node_lookup: Dict[str, ast.AST],
    source_code: str,
    embedder: Optional[SemanticEmbedder] = None,
) -> torch.Tensor:
    """Generate the feature tensor for each node in the graph."""

    node_features: list[list[float]] = []
    embedding_dim = embedder.dimension if embedder else 0

    for node_name in graph.nodes():
        base_features = get_node_features(node_name, graph)
        if embedder:
            ast_node = node_lookup.get(node_name)
            snippet = _extract_source_segment(ast_node, source_code) if ast_node else ""
            embedding = embedder.embed(node_name, snippet)
            base_features.extend(embedding)
        node_features.append(base_features)

    if not node_features:
        total_dim = 17 + embedding_dim
        return torch.zeros((0, total_dim), dtype=torch.float)

    return torch.tensor(node_features, dtype=torch.float)


def create_graph_data_from_ast(
    code_ast: ast.AST,
    source_code: str,
    *,
    embedder: Optional[SemanticEmbedder] = None,
) -> Data:
    """Produce a PyG `Data` object directly from a parsed AST."""

    graph, node_lookup = _build_graph_from_ast(code_ast, source_code)
    embedder = embedder or get_global_embedder()
    feature_tensor = _featurize_graph(graph, node_lookup, source_code, embedder)

    node_names = list(graph.nodes())
    index_lookup = {name: idx for idx, name in enumerate(node_names)}

    edge_pairs: List[Tuple[int, int]] = []
    edge_roles: List[List[str]] = []

    for source, target, attrs in graph.edges(data=True):
        edge_pairs.append((index_lookup[source], index_lookup[target]))
        edge_roles.append(list(attrs.get("edge_roles", [])))

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    edge_attr = torch.zeros((len(edge_roles), len(EDGE_ROLE_TO_INDEX)), dtype=torch.float)
    for idx, roles in enumerate(edge_roles):
        if not roles:
            continue
        for role in roles:
            role_idx = EDGE_ROLE_TO_INDEX.get(role)
            if role_idx is not None:
                edge_attr[idx, role_idx] = 1.0

    data = Data(
        x=feature_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_names=node_names,
        edge_roles=[tuple(r) for r in edge_roles],
    )

    return data


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
