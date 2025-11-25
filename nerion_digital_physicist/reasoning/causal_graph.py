"""
Causal Graph Data Structures

Defines the core data structures for representing causal relationships in code.
Used by CausalAnalyzer to build and query causal graphs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any


class NodeType(Enum):
    """Types of nodes in causal graph"""
    FUNCTION = "function"
    VARIABLE = "variable"
    EXPRESSION = "expression"
    STATEMENT = "statement"
    CONDITION = "condition"
    LOOP = "loop"
    EXCEPTION = "exception"
    IMPORT = "import"


class CausalEdgeType(Enum):
    """Types of causal relationships"""
    DATA_FLOW = "data_flow"          # Variable assignment/usage
    CONTROL_FLOW = "control_flow"    # Conditional/loop control
    FUNCTION_CALL = "function_call"  # Function invocation
    STATE_CHANGE = "state_change"    # Mutation/side effect
    EXCEPTION_FLOW = "exception"     # Exception propagation
    DEPENDENCY = "dependency"        # Import/module dependency


@dataclass
class CausalNode:
    """A node in the causal graph"""
    node_id: str
    node_type: NodeType
    name: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """A directed edge in the causal graph"""
    source_id: str
    target_id: str
    edge_type: CausalEdgeType
    mechanism: Optional[str] = None  # Human-readable explanation
    strength: float = 1.0  # Causal strength (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalGraph:
    """
    Directed graph representing causal relationships in code.

    Supports:
    - Root cause analysis (trace back from effects)
    - Impact prediction (trace forward from changes)
    - Cycle detection (circular dependencies)
    - Causal path finding
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.outgoing: Dict[str, List[CausalEdge]] = defaultdict(list)
        self.incoming: Dict[str, List[CausalEdge]] = defaultdict(list)

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        name: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        **metadata
    ) -> CausalNode:
        """Add a node to the graph"""
        node = CausalNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            file_path=file_path,
            line_number=line_number,
            metadata=metadata
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: CausalEdgeType,
        mechanism: Optional[str] = None,
        strength: float = 1.0,
        **metadata
    ) -> Optional[CausalEdge]:
        """Add an edge to the graph"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            mechanism=mechanism,
            strength=strength,
            metadata=metadata
        )
        self.edges.append(edge)
        self.outgoing[source_id].append(edge)
        self.incoming[target_id].append(edge)
        return edge

    def get_causes(self, node_id: str) -> List[CausalNode]:
        """Get immediate causes of a node (incoming edges)"""
        causes = []
        for edge in self.incoming[node_id]:
            if edge.source_id in self.nodes:
                causes.append(self.nodes[edge.source_id])
        return causes

    def get_effects(self, node_id: str) -> List[CausalNode]:
        """Get immediate effects of a node (outgoing edges)"""
        effects = []
        for edge in self.outgoing[node_id]:
            if edge.target_id in self.nodes:
                effects.append(self.nodes[edge.target_id])
        return effects

    def get_root_causes(
        self,
        node_id: str,
        max_depth: int = 10
    ) -> List[Tuple[CausalNode, int]]:
        """
        Find root causes by tracing back from node.

        Returns list of (node, distance) tuples.
        """
        root_causes = []
        visited = set()

        def trace_back(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            # Get causes
            causes = self.get_causes(current_id)

            if not causes:
                # This is a root cause
                if current_id in self.nodes:
                    root_causes.append((self.nodes[current_id], depth))
            else:
                for cause in causes:
                    trace_back(cause.node_id, depth + 1)

        trace_back(node_id, 0)
        return root_causes

    def get_all_effects(
        self,
        node_id: str,
        max_depth: int = 10
    ) -> List[Tuple[CausalNode, int]]:
        """
        Find all downstream effects by tracing forward from node.

        Returns list of (node, distance) tuples.
        """
        all_effects = []
        visited = set()

        def trace_forward(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            # Get effects
            effects = self.get_effects(current_id)

            for effect in effects:
                all_effects.append((effect, depth + 1))
                trace_forward(effect.node_id, depth + 1)

        trace_forward(node_id, 0)
        return all_effects

    def find_causal_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 10
    ) -> List[List[CausalNode]]:
        """Find all causal paths between two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []

        paths = []

        def dfs(current_id: str, path: List[CausalNode]):
            if len(path) > max_length:
                return

            if current_id == target_id:
                paths.append(path.copy())
                return

            for edge in self.outgoing[current_id]:
                if self.nodes.get(edge.target_id) and edge.target_id not in [n.node_id for n in path]:
                    path.append(self.nodes[edge.target_id])
                    dfs(edge.target_id, path)
                    path.pop()

        path = [self.nodes[source_id]]
        dfs(source_id, path)

        return paths

    def detect_cycles(self) -> List[List[CausalNode]]:
        """Detect cycles in the graph"""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node_id: str, path: List[str]):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for edge in self.outgoing[node_id]:
                target = edge.target_id
                if target not in visited:
                    dfs(target, path)
                elif target in rec_stack:
                    # Found cycle
                    cycle_start = path.index(target)
                    cycle_nodes = [self.nodes[nid] for nid in path[cycle_start:]]
                    cycles.append(cycle_nodes)

            path.pop()
            rec_stack.remove(node_id)

        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id, [])

        return cycles

    def to_pyg_edges(self) -> Tuple[List[Tuple[int, int]], List[str]]:
        """
        Convert to PyTorch Geometric edge format.

        Returns:
            (edge_pairs, edge_types) where edge_pairs is [(src_idx, tgt_idx), ...]
            and edge_types is the causal type for each edge
        """
        node_to_idx = {nid: i for i, nid in enumerate(self.nodes.keys())}
        edge_pairs = []
        edge_types = []

        for edge in self.edges:
            if edge.source_id in node_to_idx and edge.target_id in node_to_idx:
                src_idx = node_to_idx[edge.source_id]
                tgt_idx = node_to_idx[edge.target_id]
                edge_pairs.append((src_idx, tgt_idx))
                edge_types.append(edge.edge_type.value)

        return edge_pairs, edge_types


# Edge type to index mapping for one-hot encoding
CAUSAL_EDGE_TYPE_TO_INDEX: Dict[str, int] = {
    CausalEdgeType.DATA_FLOW.value: 0,
    CausalEdgeType.CONTROL_FLOW.value: 1,
    CausalEdgeType.FUNCTION_CALL.value: 2,
    CausalEdgeType.STATE_CHANGE.value: 3,
    CausalEdgeType.EXCEPTION_FLOW.value: 4,
    CausalEdgeType.DEPENDENCY.value: 5,
}
