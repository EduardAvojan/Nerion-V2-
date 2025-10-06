"""
Knowledge graph implementation for storing and retrieving code relationships.

This module provides the KnowledgeGraph class which uses NetworkX to build
and manage a directed graph representing code structure, dependencies, and
refactoring relationships.
"""
from __future__ import annotations

import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional

class KnowledgeGraph:
    """
    A graph-based memory system for storing and retrieving knowledge.
    
    This class provides methods to build and query a directed graph representing
    code structure, dependencies, and refactoring relationships.
    
    Attributes:
        graph (nx.DiGraph): The underlying NetworkX directed graph
    
    Example:
        >>> kg = KnowledgeGraph()
        >>> kg.add_node("file1.py", "File", path="/path/to/file1.py")
        >>> kg.add_edge("file1.py", "func1", "CONTAINS")
    """

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        """
        Initialize the KnowledgeGraph.
        
        Args:
            graph: Optional existing NetworkX graph to use
        """
        self.graph = graph if graph is not None else nx.DiGraph()

    def add_node(self, node_id: str, node_type: str, **attrs) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node (e.g., 'File', 'Function', 'Class')
            **attrs: Additional attributes for the node
        """
        self.graph.add_node(node_id, type=node_type, **attrs)

    def add_edge(self, source_id: str, target_id: str, edge_type: str, **attrs) -> None:
        """
        Add a directed edge to the graph.
        
        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            edge_type: Type of relationship (e.g., 'CONTAINS', 'CALLS', 'MODIFIES')
            **attrs: Additional attributes for the edge
        """
        self.graph.add_edge(source_id, target_id, type=edge_type, **attrs)

    def get_functions_in_file(self, file_id: str) -> List[str]:
        """Get all functions contained in a file."""
        return [n for n, attrs in self.graph.nodes(data=True) if attrs.get('type') == 'Function' and self.graph.has_edge(file_id, n, 'CONTAINS')]

    def get_function_calls(self, function_id: str) -> List[str]:
        """Get all functions called by a given function."""
        return [target for _, target, attrs in self.graph.out_edges(function_id, data=True) if attrs.get('type') == 'CALLS']

    def get_actions_on_file(self, file_id: str) -> List[str]:
        """Get all refactoring actions that have modified a file."""
        return [source for source, _, attrs in self.graph.in_edges(file_id, data=True) if attrs.get('type') == 'MODIFIES']

    def get_outcome_of_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get the outcome of a refactoring action."""
        for _, target, attrs in self.graph.out_edges(action_id, data=True):
            if attrs.get('type') == 'HAS_OUTCOME':
                return self.graph.nodes[target]
        return None

    def save(self, path: Path) -> None:
        """
        Save the graph to a file.
        
        Args:
            path: Path where to save the graph
        """
        nx.write_graphml(self.graph, path)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        """
        Load a graph from a file.
        
        Args:
            path: Path to the graph file
            
        Returns:
            New KnowledgeGraph instance loaded from file
        """
        graph = nx.read_graphml(path)
        return cls(graph)
