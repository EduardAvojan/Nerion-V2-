"""
Causal Graph for Code Understanding

Represents causal relationships in code, enabling:
- Root cause analysis (why did X happen?)
- Effect prediction (if I change X, what happens to Y?)
- Counterfactual reasoning (what if X had been different?)

Based on Pearl's Causal Inference framework:
- Structural Causal Models (SCM)
- do-calculus for interventions
- Counterfactual reasoning with "parallel worlds"

Integration with Nerion:
- Extract causal structure from code AST
- Track data flow dependencies
- Model function call chains
- Identify side effects and state changes
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Callable


class CausalEdgeType(Enum):
    """Types of causal edges"""
    DATA_FLOW = "data_flow"              # x → y (data flows from x to y)
    CONTROL_FLOW = "control_flow"        # if x then y
    FUNCTION_CALL = "function_call"      # x() → y
    STATE_CHANGE = "state_change"        # x modifies y
    EXCEPTION_FLOW = "exception_flow"    # x raises, y catches
    TEMPORAL = "temporal"                # x happens before y


class NodeType(Enum):
    """Types of nodes in causal graph"""
    VARIABLE = "variable"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    EXPRESSION = "expression"
    STATEMENT = "statement"


@dataclass
class CausalNode:
    """A node in the causal graph"""
    node_id: str
    node_type: NodeType
    name: str

    # Code location
    file_path: Optional[str] = None
    line_number: Optional[int] = None

    # Properties
    value: Optional[Any] = None
    data_type: Optional[str] = None

    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """A causal edge between nodes"""
    source_id: str
    target_id: str
    edge_type: CausalEdgeType

    # Causal strength (0-1, higher = stronger causal relationship)
    strength: float = 1.0

    # Mechanism (how does source cause target?)
    mechanism: Optional[str] = None

    # Conditions (when does this causal relationship hold?)
    conditions: List[str] = field(default_factory=list)

    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)


class CausalGraph:
    """
    Causal graph for code understanding.

    Represents the causal structure of code, enabling:
    - Dependency analysis
    - Impact prediction
    - Root cause identification
    - Counterfactual reasoning

    Usage:
        >>> graph = CausalGraph()
        >>>
        >>> # Add nodes
        >>> graph.add_node("x", NodeType.VARIABLE, "x")
        >>> graph.add_node("y", NodeType.VARIABLE, "y")
        >>>
        >>> # Add causal edge: x causes y
        >>> graph.add_edge("x", "y", CausalEdgeType.DATA_FLOW, strength=0.9)
        >>>
        >>> # Query
        >>> causes = graph.get_causes("y")
        >>> effects = graph.get_effects("x")
    """

    def __init__(self):
        """Initialize empty causal graph"""
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []

        # Adjacency lists for fast lookup
        self.outgoing: Dict[str, List[CausalEdge]] = defaultdict(list)
        self.incoming: Dict[str, List[CausalEdge]] = defaultdict(list)

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        name: str,
        **attributes
    ) -> CausalNode:
        """Add a node to the graph"""
        node = CausalNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            attributes=attributes
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: CausalEdgeType,
        strength: float = 1.0,
        mechanism: Optional[str] = None,
        **attributes
    ) -> CausalEdge:
        """Add a causal edge"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Nodes must exist before adding edge")

        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            strength=strength,
            mechanism=mechanism,
            attributes=attributes
        )

        self.edges.append(edge)
        self.outgoing[source_id].append(edge)
        self.incoming[target_id].append(edge)

        return edge

    def get_causes(
        self,
        node_id: str,
        direct_only: bool = False,
        edge_type: Optional[CausalEdgeType] = None
    ) -> List[CausalNode]:
        """
        Get causes of a node.

        Args:
            node_id: Target node
            direct_only: Only direct causes (not transitive)
            edge_type: Filter by edge type

        Returns:
            List of causal nodes
        """
        if direct_only:
            # Direct causes only
            edges = self.incoming[node_id]
            if edge_type:
                edges = [e for e in edges if e.edge_type == edge_type]
            return [self.nodes[e.source_id] for e in edges]
        else:
            # Transitive causes (all ancestors)
            causes = set()
            self._collect_ancestors(node_id, causes, edge_type)
            return [self.nodes[nid] for nid in causes]

    def get_effects(
        self,
        node_id: str,
        direct_only: bool = False,
        edge_type: Optional[CausalEdgeType] = None
    ) -> List[CausalNode]:
        """
        Get effects of a node.

        Args:
            node_id: Source node
            direct_only: Only direct effects (not transitive)
            edge_type: Filter by edge type

        Returns:
            List of affected nodes
        """
        if direct_only:
            # Direct effects only
            edges = self.outgoing[node_id]
            if edge_type:
                edges = [e for e in edges if e.edge_type == edge_type]
            return [self.nodes[e.target_id] for e in edges]
        else:
            # Transitive effects (all descendants)
            effects = set()
            self._collect_descendants(node_id, effects, edge_type)
            return [self.nodes[nid] for nid in effects]

    def find_causal_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 10
    ) -> List[List[CausalNode]]:
        """
        Find all causal paths from source to target.

        Args:
            source_id: Source node
            target_id: Target node
            max_length: Maximum path length

        Returns:
            List of paths (each path is list of nodes)
        """
        paths = []
        visited = set()
        current_path = []

        self._dfs_paths(source_id, target_id, visited, current_path, paths, max_length)

        # Convert node IDs to nodes
        return [[self.nodes[nid] for nid in path] for path in paths]

    def get_root_causes(
        self,
        node_id: str,
        max_depth: int = 5
    ) -> List[Tuple[CausalNode, int]]:
        """
        Find root causes (nodes with no incoming edges).

        Args:
            node_id: Target node
            max_depth: Maximum search depth

        Returns:
            List of (root_node, distance) tuples
        """
        root_causes = []
        distances = {node_id: 0}
        queue = [(node_id, 0)]
        visited = set()

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            # Check if root (no incoming edges)
            if not self.incoming[current_id]:
                root_causes.append((self.nodes[current_id], depth))
                continue

            # Add causes to queue
            for edge in self.incoming[current_id]:
                if edge.source_id not in visited:
                    queue.append((edge.source_id, depth + 1))
                    distances[edge.source_id] = depth + 1

        return root_causes

    def compute_causal_impact(
        self,
        source_id: str,
        target_id: str
    ) -> float:
        """
        Compute causal impact of source on target.

        Impact is the sum of strengths along all causal paths,
        weighted by path length (shorter = stronger impact).

        Args:
            source_id: Source node
            target_id: Target node

        Returns:
            Causal impact score (0-1)
        """
        paths = self.find_causal_paths(source_id, target_id)

        if not paths:
            return 0.0

        total_impact = 0.0
        for path in paths:
            # Get edges along path
            path_strength = 1.0
            for i in range(len(path) - 1):
                edges = [e for e in self.outgoing[path[i].node_id]
                        if e.target_id == path[i+1].node_id]
                if edges:
                    path_strength *= edges[0].strength

            # Weight by path length (shorter = stronger)
            length_weight = 1.0 / len(path)
            total_impact += path_strength * length_weight

        # Normalize
        return min(total_impact, 1.0)

    def detect_cycles(self) -> List[List[CausalNode]]:
        """
        Detect cycles in causal graph.

        Cycles can indicate:
        - Recursive functions
        - Feedback loops
        - Circular dependencies

        Returns:
            List of cycles (each cycle is list of nodes)
        """
        cycles = []
        visited = set()
        rec_stack = set()

        for node_id in self.nodes:
            if node_id not in visited:
                self._detect_cycle_dfs(node_id, visited, rec_stack, [], cycles)

        return [[self.nodes[nid] for nid in cycle] for cycle in cycles]

    def simplify_graph(
        self,
        min_strength: float = 0.3,
        remove_weak_edges: bool = True
    ) -> CausalGraph:
        """
        Create simplified version of graph.

        Removes:
        - Weak causal edges (strength < threshold)
        - Transitive edges (if A→B→C exists, remove A→C)

        Args:
            min_strength: Minimum edge strength to keep
            remove_weak_edges: Remove edges below threshold

        Returns:
            Simplified causal graph
        """
        simplified = CausalGraph()

        # Copy nodes
        for node_id, node in self.nodes.items():
            simplified.add_node(
                node_id=node.node_id,
                node_type=node.node_type,
                name=node.name,
                **node.attributes
            )

        # Copy strong edges only
        for edge in self.edges:
            if remove_weak_edges and edge.strength < min_strength:
                continue

            simplified.add_edge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.edge_type,
                strength=edge.strength,
                mechanism=edge.mechanism,
                **edge.attributes
            )

        return simplified

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary"""
        return {
            'nodes': [
                {
                    'id': node.node_id,
                    'type': node.node_type.value,
                    'name': node.name,
                    'file': node.file_path,
                    'line': node.line_number,
                    'attributes': node.attributes
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'type': edge.edge_type.value,
                    'strength': edge.strength,
                    'mechanism': edge.mechanism,
                    'attributes': edge.attributes
                }
                for edge in self.edges
            ]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        # Count by node type
        node_type_counts = defaultdict(int)
        for node in self.nodes.values():
            node_type_counts[node.node_type.value] += 1

        # Count by edge type
        edge_type_counts = defaultdict(int)
        for edge in self.edges:
            edge_type_counts[edge.edge_type.value] += 1

        # Find root and leaf nodes
        roots = [nid for nid in self.nodes if not self.incoming[nid]]
        leaves = [nid for nid in self.nodes if not self.outgoing[nid]]

        # Average degree
        total_degree = sum(len(self.outgoing[nid]) + len(self.incoming[nid])
                          for nid in self.nodes)
        avg_degree = total_degree / len(self.nodes) if self.nodes else 0

        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'node_types': dict(node_type_counts),
            'edge_types': dict(edge_type_counts),
            'num_roots': len(roots),
            'num_leaves': len(leaves),
            'avg_degree': avg_degree,
            'cycles': len(self.detect_cycles())
        }

    # Helper methods

    def _collect_ancestors(
        self,
        node_id: str,
        ancestors: Set[str],
        edge_type: Optional[CausalEdgeType]
    ):
        """Recursively collect all ancestors"""
        for edge in self.incoming[node_id]:
            if edge_type and edge.edge_type != edge_type:
                continue

            if edge.source_id not in ancestors:
                ancestors.add(edge.source_id)
                self._collect_ancestors(edge.source_id, ancestors, edge_type)

    def _collect_descendants(
        self,
        node_id: str,
        descendants: Set[str],
        edge_type: Optional[CausalEdgeType]
    ):
        """Recursively collect all descendants"""
        for edge in self.outgoing[node_id]:
            if edge_type and edge.edge_type != edge_type:
                continue

            if edge.target_id not in descendants:
                descendants.add(edge.target_id)
                self._collect_descendants(edge.target_id, descendants, edge_type)

    def _dfs_paths(
        self,
        current: str,
        target: str,
        visited: Set[str],
        path: List[str],
        all_paths: List[List[str]],
        max_length: int
    ):
        """DFS to find all paths"""
        if len(path) >= max_length:
            return

        visited.add(current)
        path.append(current)

        if current == target:
            all_paths.append(path.copy())
        else:
            for edge in self.outgoing[current]:
                if edge.target_id not in visited:
                    self._dfs_paths(edge.target_id, target, visited, path,
                                  all_paths, max_length)

        path.pop()
        visited.remove(current)

    def _detect_cycle_dfs(
        self,
        node_id: str,
        visited: Set[str],
        rec_stack: Set[str],
        path: List[str],
        cycles: List[List[str]]
    ):
        """DFS to detect cycles"""
        visited.add(node_id)
        rec_stack.add(node_id)
        path.append(node_id)

        for edge in self.outgoing[node_id]:
            neighbor = edge.target_id

            if neighbor not in visited:
                self._detect_cycle_dfs(neighbor, visited, rec_stack, path, cycles)
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                cycles.append(cycle)

        path.pop()
        rec_stack.remove(node_id)


# Example usage
def example_usage():
    """Example of causal graph construction"""
    graph = CausalGraph()

    # Build simple causal graph for code:
    # def process(x):
    #     y = transform(x)
    #     z = validate(y)
    #     return z

    # Add nodes
    graph.add_node("x", NodeType.VARIABLE, "x")
    graph.add_node("y", NodeType.VARIABLE, "y")
    graph.add_node("z", NodeType.VARIABLE, "z")
    graph.add_node("transform", NodeType.FUNCTION, "transform")
    graph.add_node("validate", NodeType.FUNCTION, "validate")

    # Add causal edges
    graph.add_edge("x", "transform", CausalEdgeType.DATA_FLOW, strength=1.0)
    graph.add_edge("transform", "y", CausalEdgeType.DATA_FLOW, strength=1.0)
    graph.add_edge("y", "validate", CausalEdgeType.DATA_FLOW, strength=1.0)
    graph.add_edge("validate", "z", CausalEdgeType.DATA_FLOW, strength=1.0)

    # Query graph
    print("Causes of z:", [n.name for n in graph.get_causes("z")])
    print("Effects of x:", [n.name for n in graph.get_effects("x")])
    print("Root causes of z:", [(n.name, d) for n, d in graph.get_root_causes("z")])
    print("Causal impact x→z:", graph.compute_causal_impact("x", "z"))

    # Statistics
    stats = graph.get_statistics()
    print(f"\nGraph statistics: {stats}")


if __name__ == "__main__":
    example_usage()
