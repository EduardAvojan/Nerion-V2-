"""
Builds a project-wide code graph representing all files, classes, functions,
and their relationships, including imports.

This module provides the ProjectParser class which analyzes Python projects
to build dependency graphs and identify test files.
"""
from __future__ import annotations

import os
import ast
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectParser:
    """
    Parses Python projects to build dependency graphs and identify test files.
    
    Attributes:
        project_root (str): Absolute path to the project root directory
        direct_index (defaultdict): Maps file paths to sets of files that import them
        all_files (set): Set of all Python files found in the project
        test_files (set): Set of test files identified by naming conventions
    """
    
    def __init__(self, project_root: str) -> None:
        """
        Initialize the ProjectParser.
        
        Args:
            project_root: Path to the project root directory
        """
        # Ensure the project root is an absolute, normalized path
        self.project_root = os.path.abspath(project_root)
        # Direct Index: Maps absolute path of a source file to a set of files that import it.
        self.direct_index = defaultdict(set)
        self.all_files = set()
        self.test_files = set()

    def parse_project(self) -> None:
        """
        Iterate through the project, parse files, and build the Direct Index.
        
        This method walks through all Python files in the project and builds
        a dependency graph showing which files import which other files.
        """
        logger.info(f"Starting project parsing at: {self.project_root}")
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self.all_files.add(file_path)
                    
                    if self._is_test_file(file_path):
                        self.test_files.add(file_path)
                    
                    # Parse every file to build the dependency index
                    self._parse_file(file_path)

        logger.info(f"Project parsing complete. Found {len(self.all_files)} Python files, including {len(self.test_files)} test files.")

    def _is_test_file(self, file_path: str) -> bool:
        """
        Determine if a file is a test file based on standard conventions.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file appears to be a test file, False otherwise
        """
        # Assuming pytest conventions (test_*.py or *_test.py)
        basename = os.path.basename(file_path)
        return basename.startswith("test_") or basename.endswith("_test.py")

    def _parse_file(self, file_path):
        """Parses a file using AST, extracts imports, and updates the Direct Index."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            tree = ast.parse(source_code, filename=file_path)
        except Exception as e:
            logger.warning(f"Error parsing file {file_path}: {e}")
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handles: import X
                for alias in node.names:
                    self._resolve_and_index(alias.name, file_path, level=0)
            elif isinstance(node, ast.ImportFrom):
                # Handles: from X import Y, from . import Y
                # node.module can be None for 'from . import Y'
                module_name = node.module if node.module else ""
                self._resolve_and_index(module_name, file_path, node.level)

    def _resolve_and_index(self, module_name, importer_path, level):
        """Resolves the import and updates the index."""
        source_path = self._resolve_import(module_name, importer_path, level)
        if source_path:
            # Map the source path (dependency) to the file that imports it (dependent)
            self.direct_index[source_path].add(importer_path)

    def _resolve_import(self, module_name, importer_path, level):
        """
        Resolves a module name to an absolute file path within the project.
        Handles absolute and relative imports, ignoring external libraries.
        """
        
        # 1. Determine the base path for the import resolution
        if level > 0:
            # Relative import (e.g., from ..utils import helper)
            base_path = os.path.dirname(importer_path)
            # Go up the directory structure based on the level (level 1 = '.', level 2 = '..')
            for _ in range(level - 1):
                base_path = os.path.dirname(base_path)
        else:
            # Absolute import. Assumed relative to the project root for internal modules.
            base_path = self.project_root

        # 2. Determine the potential path of the imported module/package
        if module_name:
            module_path_segments = module_name.split('.')
            potential_path = os.path.join(base_path, *module_path_segments)
        else:
            if level == 0:
                return None # Absolute import requires a module name
            # Handles 'from . import X'
            potential_path = base_path

        # Normalize the path to handle '..' correctly
        potential_path = os.path.normpath(potential_path)

        # 3. Verify existence and ensure it's within the project
        
        # Security check and filter for external libraries
        if not potential_path.startswith(self.project_root):
             return None

        # Case A: Module import (e.g., utils.py)
        module_file_path = potential_path + ".py"
        if os.path.exists(module_file_path):
            return module_file_path
        
        # Case B: Package import (e.g., pkg/__init__.py)
        init_py_path = os.path.join(potential_path, "__init__.py")
        if os.path.isdir(potential_path) and os.path.exists(init_py_path):
            return init_py_path

        return None

    def get_directly_affected_files(self, modified_file):
        """Returns a set of files that directly import the modified file."""
        # Ensure the input path is absolute and normalized for lookup
        if not os.path.isabs(modified_file):
             modified_file = os.path.abspath(modified_file)
        modified_file = os.path.normpath(modified_file)
        
        return self.direct_index.get(modified_file, set())

from typing import Any, Dict, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data
from .semantics import get_global_embedder

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

def main():
    """Demonstrates parsing the current project and printing the discovered structure."""
    print("Parsing project structure...")
    parser = ProjectParser(".")
    parser.discover_and_parse()

    print(f"Discovered and parsed {len(parser.parsed_files)} Python files.")

    print("\nBuilding project graph...")
    graph = parser.build_graph()

    print("Graph built successfully.")
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
