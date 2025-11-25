"""Phase 2 data pipeline: convert code graphs into PyG Data objects."""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data

from .semantics import SemanticEmbedder, get_global_embedder

LOGIC_PATH = Path(__file__).resolve().parent.parent / "environment" / "logic_v2.py"

# Edge roles captured in the graph; expand as new semantics are modelled.
# Updated to include causal edge types for causal reasoning
EDGE_ROLE_TO_INDEX: Dict[str, int] = {
    "sequence": 0,
    "call": 1,
    "shared_symbol": 2,
    "control_flow": 3,
    "data_flow": 4,
    "contains": 5,
    # Causal edge types (Phase 2 integration)
    "causal_data": 6,      # Variable causes variable (data dependency)
    "causal_control": 7,   # Condition causes branch
    "state_change": 8,     # Mutation causes state change
    "exception_flow": 9,   # Exception propagation
}


def _function_parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Set[str]:
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


def _collect_function_facts(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> Dict[str, object]:
    """Gather structural statistics and symbol usage for a function definition."""

    # Check if this node has pre-computed metrics from tree-sitter
    if hasattr(node, '_custom_metrics'):
        metrics = node._custom_metrics  # type: ignore
        return {
            "lineno": getattr(node, "lineno", 0),
            "order_index": getattr(node, "lineno", 0),  # overwritten later with ordinal position
            "line_count": float(metrics['line_count']),
            "arg_count": float(metrics['arg_count']),
            "avg_arg_length": float(metrics['avg_arg_length']),
            "docstring_length": float(metrics['docstring_length']),
            "branch_count": float(metrics['branch_count']),
            "call_count": float(metrics['call_count']),
            "return_count": float(metrics['return_count']),
            "cyclomatic_complexity": float(metrics['cyclomatic_complexity']),
            "call_targets": metrics['call_targets'],
            "reads": metrics['reads'],
            "writes": metrics['writes'],
        }

    # Otherwise, extract from Python AST
    params = _function_parameter_names(node)
    docstring = ast.get_docstring(node) or ""
    doc_len = len(docstring.strip())
    body_nodes = list(ast.walk(node))

    # Branch types - use getattr for Python 3.10+ Match support
    branch_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)
    if hasattr(ast, 'Match'):
        branch_types = (*branch_types, ast.Match)
    branch_nodes = sum(isinstance(n, branch_types) for n in body_nodes)
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

    # One-hot encoding for node types (18 types total: 2 structural + 8 statements + 8 expressions)
    features.extend(
        [
            # Structural
            1.0 if node_type == "function" else 0.0,
            1.0 if node_type == "class" else 0.0,
            # Statements
            1.0 if node_type == "if_stmt" else 0.0,
            1.0 if node_type == "for_loop" else 0.0,
            1.0 if node_type == "while_loop" else 0.0,
            1.0 if node_type == "try_block" else 0.0,
            1.0 if node_type == "assign" else 0.0,
            1.0 if node_type == "call_stmt" else 0.0,
            1.0 if node_type == "return_stmt" else 0.0,
            1.0 if node_type == "with_stmt" else 0.0,
            # Expressions
            1.0 if node_type == "call_expr" else 0.0,
            1.0 if node_type == "comparison" else 0.0,
            1.0 if node_type == "bool_op" else 0.0,
            1.0 if node_type == "bin_op" else 0.0,
            1.0 if node_type == "unary_op" else 0.0,
            1.0 if node_type == "attribute" else 0.0,
            1.0 if node_type == "comprehension" else 0.0,
            1.0 if node_type == "lambda" else 0.0,
        ]
    )

    # Node name length
    features.append(float(len(node_name)))

    # Type-specific features (13 features for all types)
    if node_type == "function":
        # Function-specific features
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
    elif node_type in ["if_stmt", "for_loop", "while_loop", "try_block", "assign", "call_stmt", "return_stmt", "with_stmt", "other_stmt"]:
        # Statement-specific features
        features.extend(
            [
                float(metadata.get("line_count", 0.0)),
                0.0,  # no arg_count
                0.0,  # no avg_arg_length
                0.0,  # no docstring
                0.0,  # no branch_count
                0.0,  # no call_count
                0.0,  # no return_count
                float(metadata.get("nested_complexity", 0.0)),
                0.0,  # no order_index
                0.0,  # no order_normalised
                0.0,  # no call_targets
                float(len(metadata.get("reads", []))),
                float(len(metadata.get("writes", []))),
            ]
        )
    elif node_type in ["call_expr", "comparison", "bool_op", "bin_op", "unary_op", "attribute", "comprehension", "lambda"]:
        # Expression-specific features
        # Use deterministic hash (multiprocessing-safe)
        operator_str = metadata.get("operator_type", "")
        operator_hash = int(hashlib.sha256(operator_str.encode('utf-8')).hexdigest()[:8], 16) % 1000
        features.extend(
            [
                0.0,  # no line_count (expressions are single line)
                float(metadata.get("arg_count", 0.0)),  # call args
                0.0,  # no avg_arg_length
                0.0,  # no docstring
                0.0,  # no branch_count
                0.0,  # no call_count
                0.0,  # no return_count
                float(operator_hash),  # operator type encoded as number
                0.0,  # no order_index
                0.0,  # no order_normalised
                0.0,  # no call_targets
                0.0,  # no reads
                0.0,  # no writes
            ]
        )
    else:
        # Unknown type: zero features
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


def _get_statement_node_type(stmt: ast.AST) -> str:
    """Determine the node type for a statement."""
    if isinstance(stmt, ast.If):
        return "if_stmt"
    elif isinstance(stmt, ast.For):
        return "for_loop"
    elif isinstance(stmt, ast.While):
        return "while_loop"
    elif isinstance(stmt, ast.Try):
        return "try_block"
    elif isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return "assign"
    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return "call_stmt"
    elif isinstance(stmt, ast.Return):
        return "return_stmt"
    elif isinstance(stmt, ast.With):
        return "with_stmt"
    else:
        return "other_stmt"


def _get_expression_node_type(expr: ast.AST) -> str:
    """Determine the node type for an expression."""
    if isinstance(expr, ast.Call):
        return "call_expr"
    elif isinstance(expr, ast.Compare):
        return "comparison"
    elif isinstance(expr, ast.BoolOp):
        return "bool_op"
    elif isinstance(expr, ast.BinOp):
        return "bin_op"
    elif isinstance(expr, ast.UnaryOp):
        return "unary_op"
    elif isinstance(expr, ast.Attribute):
        return "attribute"
    elif isinstance(expr, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
        return "comprehension"
    elif isinstance(expr, ast.Lambda):
        return "lambda"
    else:
        return None  # Skip other expressions (Name, Constant, etc.)


def _collect_statement_facts(stmt: ast.AST, source: str) -> Dict[str, object]:
    """Gather metrics for a statement node."""
    lineno = getattr(stmt, "lineno", 0)
    end_lineno = getattr(stmt, "end_lineno", lineno)
    line_count = max(1, end_lineno - lineno + 1)

    # Count nested complexity
    nested_count = 0
    for child in ast.walk(stmt):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
            nested_count += 1

    # Extract variables used
    reads: Set[str] = set()
    writes: Set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                writes.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                reads.add(node.id)

    return {
        "lineno": lineno,
        "line_count": float(line_count),
        "nested_complexity": float(nested_count),
        "reads": reads,
        "writes": writes,
    }


def _collect_expression_facts(expr: ast.AST, source: str) -> Dict[str, object]:
    """Gather metrics for an expression node."""
    lineno = getattr(expr, "lineno", 0)

    # Get operator type for comparisons and binary ops
    operator_type = ""
    if isinstance(expr, ast.Compare):
        # e.g., x > 10, data is None
        ops = expr.ops
        if ops:
            op = ops[0]
            operator_type = op.__class__.__name__  # Gt, Lt, Eq, Is, In, etc.
    elif isinstance(expr, ast.BinOp):
        operator_type = expr.op.__class__.__name__  # Add, Sub, Mult, etc.
    elif isinstance(expr, ast.UnaryOp):
        operator_type = expr.op.__class__.__name__  # Not, UAdd, USub, etc.
    elif isinstance(expr, ast.BoolOp):
        operator_type = expr.op.__class__.__name__  # And, Or
    elif isinstance(expr, ast.Call):
        # Get function name if it's a simple call
        if isinstance(expr.func, ast.Name):
            operator_type = f"call_{expr.func.id}"
        elif isinstance(expr.func, ast.Attribute):
            operator_type = f"method_{expr.func.attr}"
    elif isinstance(expr, ast.Attribute):
        operator_type = f"attr_{expr.attr}"

    # Count arguments for calls
    arg_count = 0
    if isinstance(expr, ast.Call):
        arg_count = len(expr.args) + len(expr.keywords)

    return {
        "lineno": lineno,
        "operator_type": operator_type,
        "arg_count": float(arg_count),
    }


def _build_graph_from_ast(
    code_ast: ast.AST,
    source_code: str,
) -> Tuple[nx.DiGraph, Dict[str, ast.AST]]:
    """Create a directed graph describing the code structure with statement-level granularity."""

    graph = nx.DiGraph()
    node_lookup: Dict[str, ast.AST] = {}
    metadata: Dict[str, Dict[str, object]] = {}
    stmt_counter = 0  # unique ID for statement nodes

    # First pass: Extract all functions (including async functions)
    for node in ast.walk(code_ast):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_lookup[node.name] = node
            facts = _collect_function_facts(node, source_code)
            metadata[node.name] = facts
            graph.add_node(node.name, node_type="function", metadata=facts)

    if not metadata:
        return graph, node_lookup

    # Helper to extract expressions from a statement (only direct children, not nested)
    def extract_expressions(parent_name: str, stmt: ast.stmt, expr_counter_holder: List[int]) -> None:
        """Extract important expression nodes from within a statement (non-recursive)."""
        # Only iterate through direct expression children, not nested statements
        for child in ast.iter_child_nodes(stmt):
            # Skip nested statements (they'll be handled by add_statements_recursive)
            if isinstance(child, ast.stmt):
                continue

            # Check this expression and its children
            for expr_node in ast.walk(child):
                expr_type = _get_expression_node_type(expr_node)
                if expr_type:
                    expr_name = f"{parent_name}_expr{expr_counter_holder[0]}"
                    expr_counter_holder[0] += 1

                    expr_facts = _collect_expression_facts(expr_node, source_code)
                    graph.add_node(expr_name, node_type=expr_type, metadata=expr_facts)
                    node_lookup[expr_name] = expr_node

                    # Statement contains expression
                    _add_edge_with_role(graph, parent_name, expr_name, "contains")

    # Helper function to recursively add statements
    def add_statements_recursive(parent_name: str, statements: List[ast.stmt], prev_stmt_name: Optional[str] = None) -> Optional[str]:
        """Recursively add statement nodes and return the last statement added."""
        nonlocal stmt_counter
        expr_counter = [0]  # Use list to make it mutable in nested function

        for stmt in statements:
            stmt_type = _get_statement_node_type(stmt)
            stmt_name = f"{parent_name}_stmt{stmt_counter}"
            stmt_counter += 1

            # Add statement node
            stmt_facts = _collect_statement_facts(stmt, source_code)
            graph.add_node(stmt_name, node_type=stmt_type, metadata=stmt_facts)
            node_lookup[stmt_name] = stmt

            # Parent contains statement
            _add_edge_with_role(graph, parent_name, stmt_name, "contains")

            # Sequential flow
            if prev_stmt_name is not None:
                _add_edge_with_role(graph, prev_stmt_name, stmt_name, "control_flow")

            # Extract expressions from this statement
            extract_expressions(stmt_name, stmt, expr_counter)

            # Recurse into nested blocks
            if isinstance(stmt, ast.If):
                add_statements_recursive(stmt_name, stmt.body)
                if stmt.orelse:
                    add_statements_recursive(stmt_name, stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.While)):
                add_statements_recursive(stmt_name, stmt.body)
            elif isinstance(stmt, ast.Try):
                add_statements_recursive(stmt_name, stmt.body)
                for handler in stmt.handlers:
                    add_statements_recursive(stmt_name, handler.body)
                if stmt.orelse:
                    add_statements_recursive(stmt_name, stmt.orelse)
                if stmt.finalbody:
                    add_statements_recursive(stmt_name, stmt.finalbody)
            elif isinstance(stmt, ast.With):
                add_statements_recursive(stmt_name, stmt.body)

            prev_stmt_name = stmt_name

        return prev_stmt_name

    # Second pass: Add statement-level nodes within each function
    for func_name, func_node in list(node_lookup.items()):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        add_statements_recursive(func_name, func_node.body)

    # Order functions by line number
    ordered = sorted(metadata.items(), key=lambda item: item[1].get("lineno", 0))
    denominator = float(max(len(ordered) - 1, 1))

    for ordinal, (name, facts) in enumerate(ordered):
        facts["order_index"] = float(ordinal)
        facts["order_normalised"] = float(ordinal / denominator)
        graph.nodes[name]["metadata"] = facts
        if ordinal > 0:
            prev_name = ordered[ordinal - 1][0]
            _add_edge_with_role(graph, prev_name, name, "sequence")

    # Function-to-function call edges
    defined = set(metadata.keys())
    for name, facts in metadata.items():
        for target in facts.get("call_targets", []):
            if target in defined:
                _add_edge_with_role(graph, name, target, "call")

    # Data flow edges (statement-level)
    symbol_writers: Dict[str, Set[str]] = defaultdict(set)
    symbol_readers: Dict[str, Set[str]] = defaultdict(set)

    # Collect reads/writes from all nodes (functions + statements)
    for node_name in graph.nodes():
        node_data = graph.nodes[node_name]
        node_metadata = node_data.get("metadata", {})
        for sym in node_metadata.get("writes", []):
            symbol_writers[str(sym)].add(node_name)
        for sym in node_metadata.get("reads", []):
            symbol_readers[str(sym)].add(node_name)

    # Create data flow edges
    for symbol, writers in symbol_writers.items():
        readers = symbol_readers.get(symbol, set())
        if not readers:
            continue
        for writer in writers:
            for reader in readers:
                if writer != reader:
                    _add_edge_with_role(graph, writer, reader, "data_flow")
                    # Also add causal data edge (Phase 2: Causal GNN)
                    _add_edge_with_role(graph, writer, reader, "causal_data")

    # Phase 2: Add causal edges for control structures
    for node_name in graph.nodes():
        node_data = graph.nodes[node_name]
        node_type = node_data.get("node_type", "")

        # If statements: condition causally determines branches
        if node_type == "if_stmt":
            # Find nested statements and add causal_control edges
            for target_name in graph.nodes():
                # Check if target is contained by this if
                if graph.has_edge(node_name, target_name):
                    edge_data = graph[node_name][target_name]
                    if "contains" in edge_data.get("edge_roles", []):
                        _add_edge_with_role(graph, node_name, target_name, "causal_control")

        # Assignments: cause state changes
        elif node_type == "assign":
            node_metadata = node_data.get("metadata", {})
            writes = node_metadata.get("writes", [])
            if writes:
                # Assignment causes state change
                # Find any subsequent readers of this variable
                for sym in writes:
                    readers = symbol_readers.get(str(sym), set())
                    for reader in readers:
                        if reader != node_name:
                            _add_edge_with_role(graph, node_name, reader, "state_change")

        # Try blocks: exception flow
        elif node_type == "try_block":
            for target_name in graph.nodes():
                if graph.has_edge(node_name, target_name):
                    edge_data = graph[node_name][target_name]
                    if "contains" in edge_data.get("edge_roles", []):
                        _add_edge_with_role(graph, node_name, target_name, "exception_flow")

    # Ensure metadata collections are JSON-serialisable downstream
    for node_name in graph.nodes():
        node_metadata = graph.nodes[node_name].get("metadata", {})
        if "call_targets" in node_metadata:
            node_metadata["call_targets"] = sorted(str(item) for item in node_metadata.get("call_targets", []))
        if "reads" in node_metadata:
            node_metadata["reads"] = sorted(str(item) for item in node_metadata.get("reads", []))
        if "writes" in node_metadata:
            node_metadata["writes"] = sorted(str(item) for item in node_metadata.get("writes", []))

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
        total_dim = 32 + embedding_dim  # 18 node types + 1 name length + 13 type-specific features
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
    """Produce graph data directly from source text (supports multiple languages)."""

    try:
        # Try Python AST first (fastest path)
        code_ast = ast.parse(source_code)
        return create_graph_data_from_ast(code_ast, source_code, embedder=embedder)
    except SyntaxError:
        # Fall back to multi-language parser using tree-sitter
        from .multilang_parser import convert_to_python_ast_style

        try:
            code_ast = convert_to_python_ast_style(source_code)
            return create_graph_data_from_ast(code_ast, source_code, embedder=embedder)
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Failed to parse code with both Python AST and tree-sitter: {e}") from e


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
