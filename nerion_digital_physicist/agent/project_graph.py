"""
Builds a project-wide code graph representing all files, classes, functions,
and their relationships, including imports.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data

# Directories to ignore during file discovery
IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".vscode",
    "build",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}

class ProjectParser:
    """
    Parses an entire project directory to extract information about modules,
    classes, functions, and their import relationships.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.parsed_files: Dict[str, Dict[str, Any]] = {}

    def discover_and_parse(self):
        """
        Walks the project directory, parsing all Python files.
        """
        for root, dirs, files in os.walk(self.project_root):
            # Modify dirs in-place to prune the search
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.project_root)
                    self._parse_file(file_path, str(relative_path))

    def _parse_file(self, file_path: Path, relative_path: str):
        """
        Parses a single Python file to find classes, functions, and imports.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
                tree = ast.parse(source, filename=str(file_path))
        except Exception:
            # Ignore files that can't be parsed
            return

        classes = {}
        functions = {}
        imports = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_methods = {
                    n.name: {"type": "function", "start_line": n.lineno, "end_line": n.end_lineno}
                    for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                classes[node.name] = {
                    "type": "class",
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "methods": class_methods,
                }
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = {
                    "type": "function",
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                }
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        self.parsed_files[relative_path] = {
            "path": relative_path,
            "source": source,
            "classes": classes,
            "functions": functions,
            "imports": imports,
        }

    def build_graph(self) -> nx.DiGraph:
        """
        Builds a NetworkX graph from the parsed project data.
        """
        graph = nx.DiGraph()

        # Add nodes for all modules, classes, and functions
        for file_path, data in self.parsed_files.items():
            graph.add_node(file_path, type="module")
            for class_name, class_data in data["classes"].items():
                class_node_id = f"{file_path}::{class_name}"
                graph.add_node(class_node_id, type="class")
                graph.add_edge(file_path, class_node_id, type="CONTAINS")
                for method_name in class_data["methods"]:
                    method_node_id = f"{class_node_id}::{method_name}"
                    graph.add_node(method_node_id, type="function")
                    graph.add_edge(class_node_id, method_node_id, type="CONTAINS")
            for func_name in data["functions"]:
                func_node_id = f"{file_path}::{func_name}"
                graph.add_node(func_node_id, type="function")
                graph.add_edge(file_path, func_node_id, type="CONTAINS")

        # Add import edges
        for file_path, data in self.parsed_files.items():
            for imp in data["imports"]:
                # This is a simplified resolver. A real one would be more complex.
                resolved_path = self._resolve_import(imp, Path(file_path).parent)
                if resolved_path and graph.has_node(resolved_path):
                    graph.add_edge(file_path, resolved_path, type="IMPORTS")

        return graph

    def _resolve_import(self, import_name: str, current_dir: Path) -> str | None:
        """
        Resolves an import name to a relative file path.
        """
        try:
            # Try to resolve as a file (e.g., `from . import foo` -> `foo.py`)
            potential_path = (current_dir / f"{import_name}.py").relative_to(self.project_root)
            if str(potential_path) in self.parsed_files:
                return str(potential_path)

            # Try to resolve as a package/directory (e.g., `import foo` -> `foo/__init__.py`)
            potential_path = (current_dir / import_name / "__init__.py").relative_to(self.project_root)
            if str(potential_path) in self.parsed_files:
                return str(potential_path)
        except ValueError:
            # This happens if the path is not within the project root (e.g., a stdlib import)
            pass
        
        # Try absolute import from project root
        abs_path_str = import_name.replace(".", "/")
        potential_path_file = Path(f"{abs_path_str}.py")
        if str(potential_path_file) in self.parsed_files:
            return str(potential_path_file)
        
        potential_path_pkg = Path(f"{abs_path_str}/__init__.py")
        if str(potential_path_pkg) in self.parsed_files:
            return str(potential_path_pkg)

        return None

from .semantics import get_global_embedder

def _extract_node_snippet(node_id: str, node_data: dict, parser: ProjectParser) -> Tuple[str, str]:
    """Extracts a name and code snippet for a given node."""
    node_type = node_data.get("type", "unknown")
    parts = node_id.split("::")
    file_path = parts[0]
    source = parser.parsed_files.get(file_path, {}).get("source", "")
    lines = source.splitlines()

    if node_type == "module":
        return parts[0], source
    elif node_type == "class":
        class_name = parts[1]
        class_info = parser.parsed_files.get(file_path, {}).get("classes", {}).get(class_name, {})
        start, end = class_info.get("start_line", 1) - 1, class_info.get("end_line", 1)
        return class_name, "\n".join(lines[start:end])
    elif node_type == "function":
        func_name = parts[-1]
        if len(parts) == 3: # Method
            class_name = parts[1]
            func_info = parser.parsed_files.get(file_path, {}).get("classes", {}).get(class_name, {}).get("methods", {}).get(func_name, {})
        else: # Top-level function
            func_info = parser.parsed_files.get(file_path, {}).get("functions", {}).get(func_name, {})
        start, end = func_info.get("start_line", 1) - 1, func_info.get("end_line", 1)
        return func_name, "\n".join(lines[start:end])
    return "", ""

def to_pyg_data(graph: nx.DiGraph, parser: ProjectParser) -> Tuple[Data, Dict[str, int]]:
    """
    Converts the NetworkX graph to a PyTorch Geometric Data object with rich features.
    Also returns the mapping from node ID to integer index.
    """
    embedder = get_global_embedder()
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    # Create a reverse map from index to the full node data for the harness
    node_map_reverse = {i: {"id": node_id, "type": graph.nodes[node_id].get("type"), "name": _extract_node_snippet(node_id, graph.nodes[node_id], parser)[0]} for node_id, i in node_mapping.items()}

    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in graph.edges()], dtype=torch.long).t().contiguous()

    node_features = []
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get("type", "unknown")
        
        # One-hot encoding for type
        type_feature = [
            1 if node_type == "module" else 0,
            1 if node_type == "class" else 0,
            1 if node_type == "function" else 0,
        ]

        # Semantic embedding
        name, snippet = _extract_node_snippet(node_id, node_data, parser)
        embedding = embedder.embed(name, snippet)

        # Combine features
        node_features.append(type_feature + embedding)

    x = torch.tensor(node_features, dtype=torch.float)
    
    # The harness needs the reverse map to find nodes by name/path
    return Data(x=x, edge_index=edge_index), node_map_reverse

def main():
    """
    Demonstrates parsing the current project and printing the discovered structure.
    """
    print("Parsing project structure...")
    parser = ProjectParser(".")
    parser.discover_and_parse()

    print(f"Discovered and parsed {len(parser.parsed_files)} Python files.")

    print("\nBuilding project graph...")
    graph = parser.build_graph()

    print(f"Graph built successfully.")
    print(f"  - Total nodes: {graph.number_of_nodes()}")
    print(f"  - Total edges: {graph.number_of_edges()}")

    # Count node types
    node_types = [data["type"] for _, data in graph.nodes(data=True)]
    print(f"  - Module nodes: {node_types.count('module')}")
    print(f"  - Class nodes: {node_types.count('class')}")
    print(f"  - Function nodes: {node_types.count('function')}")

    print("\nConverting to PyTorch Geometric data object with semantic features...")
    pyg_data = to_pyg_data(graph, parser)

    print("Conversion successful.")
    print(pyg_data)


if __name__ == "__main__":
    # Add parent to path to allow running as a script
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    main()
