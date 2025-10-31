"""
Counterfactual Reasoning for Code

Enables "what-if" reasoning about code:
- What if this variable had a different value?
- What if this function was never called?
- What if this condition evaluated differently?

Based on Pearl's Counterfactual Inference:
1. Abduction: Infer what must have been true
2. Action: Modify causal graph (intervention)
3. Prediction: Predict outcome in modified world

Integration with Nerion:
- Uses causal graphs from CausalAnalyzer
- Enables impact prediction before code changes
- Supports root cause analysis
- Feeds into ChainOfThoughtReasoner for consequence prediction
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable

from nerion_digital_physicist.reasoning.causal_graph import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    CausalEdgeType,
    NodeType
)


class InterventionType(Enum):
    """Types of interventions"""
    SET_VALUE = "set_value"              # Set variable to specific value
    REMOVE_NODE = "remove_node"          # Remove node (e.g., delete code)
    ADD_EDGE = "add_edge"                # Add causal dependency
    REMOVE_EDGE = "remove_edge"          # Remove causal dependency
    CHANGE_EDGE = "change_edge"          # Modify edge strength


@dataclass
class Intervention:
    """An intervention on the causal graph"""
    intervention_type: InterventionType
    target_node_id: str

    # For SET_VALUE
    new_value: Optional[Any] = None

    # For ADD_EDGE
    source_node_id: Optional[str] = None
    edge_type: Optional[CausalEdgeType] = None

    # For CHANGE_EDGE
    new_strength: Optional[float] = None

    # Description
    description: str = ""


@dataclass
class CounterfactualQuery:
    """A counterfactual "what-if" query"""
    query_description: str
    interventions: List[Intervention]
    target_variables: List[str]  # Variables to predict outcome for


@dataclass
class CounterfactualResult:
    """Result of counterfactual reasoning"""
    query: CounterfactualQuery
    predicted_outcomes: Dict[str, Any]  # variable -> predicted value
    confidence: float
    causal_paths_affected: List[List[CausalNode]]
    explanation: str


class CounterfactualReasoner:
    """
    Counterfactual reasoning engine for code.

    Performs "what-if" analysis by:
    1. Abduction: Understand current state
    2. Intervention: Modify causal graph
    3. Prediction: Predict outcomes in modified world

    Usage:
        >>> reasoner = CounterfactualReasoner(causal_graph)
        >>>
        >>> # What if variable x had value 10?
        >>> query = CounterfactualQuery(
        ...     query_description="What if x = 10?",
        ...     interventions=[Intervention(
        ...         intervention_type=InterventionType.SET_VALUE,
        ...         target_node_id="var_x",
        ...         new_value=10
        ...     )],
        ...     target_variables=["y", "z"]
        ... )
        >>>
        >>> result = reasoner.reason(query)
        >>> print(result.explanation)
    """

    def __init__(self, causal_graph: CausalGraph):
        """
        Initialize counterfactual reasoner.

        Args:
            causal_graph: Causal graph of code
        """
        self.original_graph = causal_graph

    def reason(
        self,
        query: CounterfactualQuery,
        current_values: Optional[Dict[str, Any]] = None
    ) -> CounterfactualResult:
        """
        Perform counterfactual reasoning.

        Args:
            query: Counterfactual query
            current_values: Current variable values (for abduction)

        Returns:
            CounterfactualResult with predictions
        """
        print(f"[Counterfactual] Reasoning about: {query.query_description}")

        # Step 1: Abduction (understand current state)
        current_state = self._abduction(current_values or {})

        # Step 2: Action (apply interventions)
        modified_graph = self._apply_interventions(query.interventions)

        # Step 3: Prediction (predict outcomes in modified world)
        predicted_outcomes, confidence = self._predict_outcomes(
            modified_graph,
            current_state,
            query.target_variables
        )

        # Analyze affected causal paths
        affected_paths = self._analyze_affected_paths(
            query.interventions,
            query.target_variables,
            modified_graph
        )

        # Generate explanation
        explanation = self._generate_explanation(
            query,
            predicted_outcomes,
            affected_paths,
            confidence
        )

        result = CounterfactualResult(
            query=query,
            predicted_outcomes=predicted_outcomes,
            confidence=confidence,
            causal_paths_affected=affected_paths,
            explanation=explanation
        )

        print(f"[Counterfactual] Confidence: {confidence:.0%}")
        return result

    def compare_interventions(
        self,
        interventions_list: List[List[Intervention]],
        target_variable: str,
        current_values: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[List[Intervention], Any, float]]:
        """
        Compare multiple intervention scenarios.

        Args:
            interventions_list: List of intervention scenarios
            target_variable: Variable to compare outcomes for
            current_values: Current state

        Returns:
            List of (interventions, predicted_outcome, confidence) tuples
        """
        results = []

        for interventions in interventions_list:
            query = CounterfactualQuery(
                query_description=f"Scenario {len(results) + 1}",
                interventions=interventions,
                target_variables=[target_variable]
            )

            result = self.reason(query, current_values)
            predicted_value = result.predicted_outcomes.get(target_variable)

            results.append((interventions, predicted_value, result.confidence))

        # Sort by predicted outcome (assuming higher is better)
        results.sort(key=lambda x: (x[1] if x[1] is not None else 0), reverse=True)

        return results

    def _abduction(self, current_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abduction: Infer what must be true given observations.

        In code context, this means understanding current variable values
        and intermediate states.

        Args:
            current_values: Observed variable values

        Returns:
            Inferred complete state
        """
        inferred_state = current_values.copy()

        # Infer values for unobserved variables
        # (Simplified - would use constraint solving in production)
        for node_id, node in self.original_graph.nodes.items():
            if node.name not in inferred_state and node.node_type == NodeType.VARIABLE:
                # Try to infer from causes
                causes = self.original_graph.get_causes(node_id, direct_only=True)

                if causes and all(c.name in inferred_state for c in causes):
                    # Simple inference: aggregate cause values
                    cause_values = [inferred_state[c.name] for c in causes]
                    # Placeholder inference (would use actual semantics)
                    inferred_state[node.name] = cause_values[0] if cause_values else None

        return inferred_state

    def _apply_interventions(
        self,
        interventions: List[Intervention]
    ) -> CausalGraph:
        """
        Apply interventions to create modified causal graph.

        Creates a copy of original graph and applies modifications.

        Args:
            interventions: List of interventions to apply

        Returns:
            Modified causal graph
        """
        # Deep copy graph
        modified_graph = CausalGraph()

        # Copy nodes
        for node_id, node in self.original_graph.nodes.items():
            modified_graph.add_node(
                node_id=node.node_id,
                node_type=node.node_type,
                name=node.name,
                file_path=node.file_path,
                line_number=node.line_number,
                **node.attributes
            )

        # Copy edges
        for edge in self.original_graph.edges:
            modified_graph.add_edge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.edge_type,
                strength=edge.strength,
                mechanism=edge.mechanism,
                **edge.attributes
            )

        # Apply interventions
        for intervention in interventions:
            if intervention.intervention_type == InterventionType.SET_VALUE:
                # Setting value cuts incoming edges (do-operator)
                target_id = intervention.target_node_id
                if target_id in modified_graph.nodes:
                    # Remove incoming edges (intervention breaks causal dependencies)
                    modified_graph.incoming[target_id] = []

                    # Update node value
                    modified_graph.nodes[target_id].value = intervention.new_value

            elif intervention.intervention_type == InterventionType.REMOVE_NODE:
                # Remove node and its edges
                target_id = intervention.target_node_id
                if target_id in modified_graph.nodes:
                    # Remove outgoing edges
                    for edge in modified_graph.outgoing[target_id]:
                        modified_graph.edges.remove(edge)
                        modified_graph.incoming[edge.target_id].remove(edge)

                    # Remove incoming edges
                    for edge in modified_graph.incoming[target_id]:
                        modified_graph.edges.remove(edge)
                        modified_graph.outgoing[edge.source_id].remove(edge)

                    # Remove node
                    del modified_graph.nodes[target_id]
                    del modified_graph.outgoing[target_id]
                    del modified_graph.incoming[target_id]

            elif intervention.intervention_type == InterventionType.ADD_EDGE:
                # Add new causal edge
                if intervention.source_node_id and intervention.edge_type:
                    modified_graph.add_edge(
                        source_id=intervention.source_node_id,
                        target_id=intervention.target_node_id,
                        edge_type=intervention.edge_type,
                        strength=1.0,
                        mechanism="counterfactual_intervention"
                    )

            elif intervention.intervention_type == InterventionType.REMOVE_EDGE:
                # Remove specific edge
                edges_to_remove = [
                    e for e in modified_graph.edges
                    if e.target_id == intervention.target_node_id and
                       (intervention.source_node_id is None or
                        e.source_id == intervention.source_node_id)
                ]
                for edge in edges_to_remove:
                    modified_graph.edges.remove(edge)
                    modified_graph.outgoing[edge.source_id].remove(edge)
                    modified_graph.incoming[edge.target_id].remove(edge)

            elif intervention.intervention_type == InterventionType.CHANGE_EDGE:
                # Modify edge strength
                for edge in modified_graph.outgoing.get(intervention.target_node_id, []):
                    if intervention.new_strength is not None:
                        edge.strength = intervention.new_strength

        return modified_graph

    def _predict_outcomes(
        self,
        modified_graph: CausalGraph,
        current_state: Dict[str, Any],
        target_variables: List[str]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Predict outcomes in modified causal graph.

        Args:
            modified_graph: Graph after interventions
            current_state: Current variable values
            target_variables: Variables to predict

        Returns:
            (predicted_outcomes, confidence)
        """
        predicted = {}
        confidences = []

        for target_var in target_variables:
            # Find node for target variable
            target_node_id = None
            for node_id, node in modified_graph.nodes.items():
                if node.name == target_var:
                    target_node_id = node_id
                    break

            if not target_node_id:
                predicted[target_var] = None
                confidences.append(0.0)
                continue

            # Check if value already set by intervention
            if modified_graph.nodes[target_node_id].value is not None:
                predicted[target_var] = modified_graph.nodes[target_node_id].value
                confidences.append(1.0)
                continue

            # Predict based on causes
            causes = modified_graph.get_causes(target_node_id, direct_only=True)

            if not causes:
                # No causes, use current value
                predicted[target_var] = current_state.get(target_var)
                confidences.append(0.5)
            else:
                # Aggregate cause values (simplified)
                cause_values = []
                confidence = 1.0

                for cause in causes:
                    if cause.name in current_state:
                        cause_values.append(current_state[cause.name])
                    elif cause.value is not None:
                        cause_values.append(cause.value)
                    else:
                        confidence *= 0.7  # Reduce confidence for unknown causes

                # Simple prediction (would use actual semantics)
                if cause_values:
                    predicted[target_var] = cause_values[0]
                else:
                    predicted[target_var] = None
                    confidence = 0.3

                confidences.append(confidence)

        # Overall confidence
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return predicted, overall_confidence

    def _analyze_affected_paths(
        self,
        interventions: List[Intervention],
        target_variables: List[str],
        modified_graph: CausalGraph
    ) -> List[List[CausalNode]]:
        """
        Identify causal paths affected by interventions.

        Args:
            interventions: Applied interventions
            target_variables: Target variables
            modified_graph: Modified graph

        Returns:
            List of affected causal paths
        """
        affected_paths = []

        for intervention in interventions:
            for target_var in target_variables:
                # Find node IDs
                intervention_node = intervention.target_node_id

                target_node_id = None
                for node_id, node in modified_graph.nodes.items():
                    if node.name == target_var:
                        target_node_id = node_id
                        break

                if not target_node_id:
                    continue

                # Find paths from intervention to target
                paths = modified_graph.find_causal_paths(
                    intervention_node,
                    target_node_id,
                    max_length=5
                )

                affected_paths.extend(paths)

        return affected_paths

    def _generate_explanation(
        self,
        query: CounterfactualQuery,
        predicted_outcomes: Dict[str, Any],
        affected_paths: List[List[CausalNode]],
        confidence: float
    ) -> str:
        """Generate human-readable explanation"""
        explanation = f"## Counterfactual Analysis: {query.query_description}\n\n"

        # Interventions
        explanation += "**Interventions:**\n"
        for i, intervention in enumerate(query.interventions, 1):
            explanation += f"{i}. {intervention.description or intervention.intervention_type.value}\n"

        # Predicted outcomes
        explanation += "\n**Predicted Outcomes:**\n"
        for var, value in predicted_outcomes.items():
            explanation += f"- `{var}` → {value}\n"

        # Confidence
        explanation += f"\n**Confidence:** {confidence:.0%}\n"

        # Affected paths
        if affected_paths:
            explanation += "\n**Causal Paths Affected:**\n"
            for path in affected_paths[:3]:  # Show top 3
                path_str = " → ".join(node.name for node in path)
                explanation += f"- {path_str}\n"

        return explanation


# Helper functions for common counterfactual queries

def what_if_variable_had_value(
    causal_graph: CausalGraph,
    variable_name: str,
    new_value: Any,
    target_variables: List[str]
) -> CounterfactualResult:
    """
    What if variable had a different value?

    Args:
        causal_graph: Causal graph
        variable_name: Variable to modify
        new_value: New value to set
        target_variables: Variables to predict outcome for

    Returns:
        CounterfactualResult
    """
    # Find node for variable
    var_node_id = None
    for node_id, node in causal_graph.nodes.items():
        if node.name == variable_name:
            var_node_id = node_id
            break

    if not var_node_id:
        raise ValueError(f"Variable {variable_name} not found in graph")

    reasoner = CounterfactualReasoner(causal_graph)

    query = CounterfactualQuery(
        query_description=f"What if {variable_name} = {new_value}?",
        interventions=[
            Intervention(
                intervention_type=InterventionType.SET_VALUE,
                target_node_id=var_node_id,
                new_value=new_value,
                description=f"Set {variable_name} to {new_value}"
            )
        ],
        target_variables=target_variables
    )

    return reasoner.reason(query)


def what_if_function_not_called(
    causal_graph: CausalGraph,
    function_name: str,
    target_variables: List[str]
) -> CounterfactualResult:
    """
    What if function was never called?

    Args:
        causal_graph: Causal graph
        function_name: Function to remove
        target_variables: Variables to predict outcome for

    Returns:
        CounterfactualResult
    """
    # Find node for function
    func_node_id = None
    for node_id, node in causal_graph.nodes.items():
        if node.name == function_name and node.node_type == NodeType.FUNCTION:
            func_node_id = node_id
            break

    if not func_node_id:
        raise ValueError(f"Function {function_name} not found in graph")

    reasoner = CounterfactualReasoner(causal_graph)

    query = CounterfactualQuery(
        query_description=f"What if {function_name}() was never called?",
        interventions=[
            Intervention(
                intervention_type=InterventionType.REMOVE_NODE,
                target_node_id=func_node_id,
                description=f"Remove {function_name}() call"
            )
        ],
        target_variables=target_variables
    )

    return reasoner.reason(query)


# Example usage
def example_usage():
    """Example of counterfactual reasoning"""
    from nerion_digital_physicist.reasoning.causal_graph import CausalGraph, NodeType, CausalEdgeType

    # Build simple causal graph
    graph = CausalGraph()
    graph.add_node("x", NodeType.VARIABLE, "x")
    graph.add_node("y", NodeType.VARIABLE, "y")
    graph.add_node("z", NodeType.VARIABLE, "z")

    graph.add_edge("x", "y", CausalEdgeType.DATA_FLOW, strength=0.9)
    graph.add_edge("y", "z", CausalEdgeType.DATA_FLOW, strength=0.8)

    # Counterfactual: What if x = 100?
    result = what_if_variable_had_value(
        causal_graph=graph,
        variable_name="x",
        new_value=100,
        target_variables=["y", "z"]
    )

    print(result.explanation)


if __name__ == "__main__":
    example_usage()
