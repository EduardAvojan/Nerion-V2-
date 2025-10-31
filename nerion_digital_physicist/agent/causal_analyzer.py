"""
Causal Analyzer for Code

Extracts causal graphs from code AST to enable:
- Root cause analysis for bugs
- Impact prediction for changes
- Dependency understanding
- Test outcome prediction

Integration with Nerion:
- Extends existing AST analysis
- Connects to ChainOfThoughtReasoner for causal prediction
- Feeds into Planner for impact-aware planning
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

from nerion_digital_physicist.reasoning.causal_graph import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    CausalEdgeType,
    NodeType
)


@dataclass
class CausalAnalysisResult:
    """Result of causal analysis"""
    graph: CausalGraph
    root_causes: List[Tuple[CausalNode, int]]  # (node, distance)
    critical_nodes: List[CausalNode]            # High-impact nodes
    bottlenecks: List[CausalNode]               # Nodes with many dependencies
    cycles: List[List[CausalNode]]              # Circular dependencies


class CausalAnalyzer:
    """
    Extracts causal graphs from code AST.

    Analyzes code to build causal graph showing:
    - Data flow (variable assignments, function arguments)
    - Control flow (conditionals, loops)
    - Function calls
    - State changes (mutations, side effects)

    Usage:
        >>> analyzer = CausalAnalyzer()
        >>> result = analyzer.analyze_code(source_code)
        >>> print(f"Root causes: {result.root_causes}")
        >>> print(f"Critical nodes: {result.critical_nodes}")
    """

    def __init__(self):
        """Initialize causal analyzer"""
        self.graph: Optional[CausalGraph] = None
        self.var_defs: Dict[str, str] = {}  # var_name -> node_id
        self.func_defs: Dict[str, str] = {}  # func_name -> node_id

    def analyze_code(
        self,
        source_code: str,
        file_path: Optional[str] = None
    ) -> CausalAnalysisResult:
        """
        Analyze code to extract causal graph.

        Args:
            source_code: Python source code
            file_path: Optional file path for metadata

        Returns:
            CausalAnalysisResult with graph and analysis
        """
        print(f"[CausalAnalyzer] Analyzing code...")

        # Initialize graph
        self.graph = CausalGraph()
        self.var_defs = {}
        self.func_defs = {}

        # Parse AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"[CausalAnalyzer] Syntax error: {e}")
            return CausalAnalysisResult(
                graph=self.graph,
                root_causes=[],
                critical_nodes=[],
                bottlenecks=[],
                cycles=[]
            )

        # Extract causal structure
        self._visit_node(tree, file_path)

        # Analyze graph
        root_causes = self._find_root_causes()
        critical_nodes = self._find_critical_nodes()
        bottlenecks = self._find_bottlenecks()
        cycles = self.graph.detect_cycles()

        print(f"[CausalAnalyzer] Extracted graph: "
              f"{len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        print(f"[CausalAnalyzer] Found: {len(root_causes)} root causes, "
              f"{len(critical_nodes)} critical nodes, {len(cycles)} cycles")

        return CausalAnalysisResult(
            graph=self.graph,
            root_causes=root_causes,
            critical_nodes=critical_nodes,
            bottlenecks=bottlenecks,
            cycles=cycles
        )

    def predict_test_outcome(
        self,
        test_code: str,
        causal_result: CausalAnalysisResult
    ) -> Tuple[str, float]:
        """
        Predict test outcome before running.

        Analyzes causal paths from test inputs to assertions
        to predict whether test will pass or fail.

        Args:
            test_code: Test code to analyze
            causal_result: Causal analysis of code under test

        Returns:
            (prediction, confidence) where prediction is "pass" or "fail"
        """
        # Parse test code
        try:
            test_tree = ast.parse(test_code)
        except SyntaxError:
            return ("unknown", 0.0)

        # Find assertions
        assertions = self._find_assertions(test_tree)

        if not assertions:
            return ("unknown", 0.0)

        # Analyze each assertion
        predictions = []
        for assertion in assertions:
            # Extract variables involved in assertion
            involved_vars = self._extract_vars_from_assertion(assertion)

            # Check if any involved variables have concerning causal paths
            has_risk = self._check_causal_risk(involved_vars, causal_result.graph)

            if has_risk:
                predictions.append("fail")
            else:
                predictions.append("pass")

        # Aggregate predictions
        fail_count = predictions.count("fail")
        pass_count = predictions.count("pass")

        if fail_count > pass_count:
            prediction = "fail"
            confidence = fail_count / len(predictions)
        else:
            prediction = "pass"
            confidence = pass_count / len(predictions)

        return (prediction, confidence)

    def identify_root_cause(
        self,
        error_variable: str,
        causal_result: CausalAnalysisResult,
        max_depth: int = 5
    ) -> List[Tuple[CausalNode, int, str]]:
        """
        Identify root cause of error.

        Traces back from error variable to find root causes.

        Args:
            error_variable: Variable where error manifests
            causal_result: Causal analysis result
            max_depth: Maximum search depth

        Returns:
            List of (root_cause_node, distance, explanation) tuples
        """
        # Find node for error variable
        error_node_id = None
        for node_id, node in causal_result.graph.nodes.items():
            if node.name == error_variable:
                error_node_id = node_id
                break

        if not error_node_id:
            return []

        # Get root causes
        root_causes = causal_result.graph.get_root_causes(error_node_id, max_depth)

        # Generate explanations
        results = []
        for root_node, distance in root_causes:
            # Find causal path
            paths = causal_result.graph.find_causal_paths(
                root_node.node_id,
                error_node_id
            )

            if paths:
                path = paths[0]  # Use first path
                explanation = self._explain_causal_path(path)
            else:
                explanation = f"Root cause: {root_node.name}"

            results.append((root_node, distance, explanation))

        return results

    def _visit_node(self, node: ast.AST, file_path: Optional[str]):
        """Visit AST node and extract causal information"""
        if isinstance(node, ast.FunctionDef):
            self._handle_function_def(node, file_path)
        elif isinstance(node, ast.Assign):
            self._handle_assignment(node, file_path)
        elif isinstance(node, ast.AugAssign):
            self._handle_aug_assignment(node, file_path)
        elif isinstance(node, ast.Call):
            self._handle_function_call(node, file_path)
        elif isinstance(node, ast.If):
            self._handle_conditional(node, file_path)
        elif isinstance(node, ast.Return):
            self._handle_return(node, file_path)

        # Recurse on children
        for child in ast.iter_child_nodes(node):
            self._visit_node(child, file_path)

    def _handle_function_def(self, node: ast.FunctionDef, file_path: Optional[str]):
        """Handle function definition"""
        func_id = f"func_{node.name}_{node.lineno}"

        # Add function node
        self.graph.add_node(
            node_id=func_id,
            node_type=NodeType.FUNCTION,
            name=node.name,
            file_path=file_path,
            line_number=node.lineno
        )

        self.func_defs[node.name] = func_id

        # Add parameter nodes
        for arg in node.args.args:
            param_id = f"param_{arg.arg}_{node.lineno}"
            self.graph.add_node(
                node_id=param_id,
                node_type=NodeType.VARIABLE,
                name=arg.arg,
                file_path=file_path,
                line_number=node.lineno
            )

            # Parameter flows into function
            self.graph.add_edge(
                source_id=param_id,
                target_id=func_id,
                edge_type=CausalEdgeType.DATA_FLOW,
                mechanism="function_parameter"
            )

    def _handle_assignment(self, node: ast.Assign, file_path: Optional[str]):
        """Handle variable assignment"""
        # Get target variable(s)
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_id = f"var_{var_name}_{node.lineno}"

                # Add variable node
                self.graph.add_node(
                    node_id=var_id,
                    node_type=NodeType.VARIABLE,
                    name=var_name,
                    file_path=file_path,
                    line_number=node.lineno
                )

                self.var_defs[var_name] = var_id

                # Extract dependencies from value
                deps = self._extract_dependencies(node.value)

                # Add causal edges from dependencies
                for dep in deps:
                    if dep in self.var_defs:
                        self.graph.add_edge(
                            source_id=self.var_defs[dep],
                            target_id=var_id,
                            edge_type=CausalEdgeType.DATA_FLOW,
                            mechanism="assignment"
                        )

    def _handle_aug_assignment(self, node: ast.AugAssign, file_path: Optional[str]):
        """Handle augmented assignment (x += y)"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id

            if var_name in self.var_defs:
                old_var_id = self.var_defs[var_name]
                new_var_id = f"var_{var_name}_{node.lineno}"

                # Add new state node
                self.graph.add_node(
                    node_id=new_var_id,
                    node_type=NodeType.VARIABLE,
                    name=var_name,
                    file_path=file_path,
                    line_number=node.lineno
                )

                # Old state causes new state
                self.graph.add_edge(
                    source_id=old_var_id,
                    target_id=new_var_id,
                    edge_type=CausalEdgeType.STATE_CHANGE,
                    mechanism="augmented_assignment"
                )

                # Dependencies from value
                deps = self._extract_dependencies(node.value)
                for dep in deps:
                    if dep in self.var_defs:
                        self.graph.add_edge(
                            source_id=self.var_defs[dep],
                            target_id=new_var_id,
                            edge_type=CausalEdgeType.DATA_FLOW
                        )

                self.var_defs[var_name] = new_var_id

    def _handle_function_call(self, node: ast.Call, file_path: Optional[str]):
        """Handle function call"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            call_id = f"call_{func_name}_{node.lineno}"

            # Add call node
            self.graph.add_node(
                node_id=call_id,
                node_type=NodeType.EXPRESSION,
                name=f"{func_name}()",
                file_path=file_path,
                line_number=node.lineno
            )

            # If function defined, add edge
            if func_name in self.func_defs:
                self.graph.add_edge(
                    source_id=self.func_defs[func_name],
                    target_id=call_id,
                    edge_type=CausalEdgeType.FUNCTION_CALL
                )

            # Arguments flow into call
            for arg in node.args:
                deps = self._extract_dependencies(arg)
                for dep in deps:
                    if dep in self.var_defs:
                        self.graph.add_edge(
                            source_id=self.var_defs[dep],
                            target_id=call_id,
                            edge_type=CausalEdgeType.DATA_FLOW,
                            mechanism="function_argument"
                        )

    def _handle_conditional(self, node: ast.If, file_path: Optional[str]):
        """Handle conditional statement"""
        cond_id = f"cond_{node.lineno}"

        # Add condition node
        self.graph.add_node(
            node_id=cond_id,
            node_type=NodeType.EXPRESSION,
            name="if_condition",
            file_path=file_path,
            line_number=node.lineno
        )

        # Extract variables from condition
        deps = self._extract_dependencies(node.test)
        for dep in deps:
            if dep in self.var_defs:
                self.graph.add_edge(
                    source_id=self.var_defs[dep],
                    target_id=cond_id,
                    edge_type=CausalEdgeType.CONTROL_FLOW,
                    mechanism="conditional_test"
                )

    def _handle_return(self, node: ast.Return, file_path: Optional[str]):
        """Handle return statement"""
        if node.value:
            ret_id = f"return_{node.lineno}"

            self.graph.add_node(
                node_id=ret_id,
                node_type=NodeType.STATEMENT,
                name="return",
                file_path=file_path,
                line_number=node.lineno
            )

            # Extract dependencies
            deps = self._extract_dependencies(node.value)
            for dep in deps:
                if dep in self.var_defs:
                    self.graph.add_edge(
                        source_id=self.var_defs[dep],
                        target_id=ret_id,
                        edge_type=CausalEdgeType.DATA_FLOW,
                        mechanism="return_value"
                    )

    def _extract_dependencies(self, node: ast.AST) -> Set[str]:
        """Extract variable dependencies from expression"""
        deps = set()

        if isinstance(node, ast.Name):
            deps.add(node.id)
        elif isinstance(node, ast.BinOp):
            deps.update(self._extract_dependencies(node.left))
            deps.update(self._extract_dependencies(node.right))
        elif isinstance(node, ast.UnaryOp):
            deps.update(self._extract_dependencies(node.operand))
        elif isinstance(node, ast.Compare):
            deps.update(self._extract_dependencies(node.left))
            for comp in node.comparators:
                deps.update(self._extract_dependencies(comp))
        elif isinstance(node, ast.Call):
            for arg in node.args:
                deps.update(self._extract_dependencies(arg))

        return deps

    def _find_root_causes(self) -> List[Tuple[CausalNode, int]]:
        """Find root cause nodes (no incoming edges)"""
        root_causes = []
        for node_id, node in self.graph.nodes.items():
            if not self.graph.incoming[node_id]:
                root_causes.append((node, 0))
        return root_causes

    def _find_critical_nodes(self) -> List[CausalNode]:
        """Find critical nodes (high impact)"""
        critical = []
        for node_id, node in self.graph.nodes.items():
            # Critical if has many effects
            effects = self.graph.get_effects(node_id)
            if len(effects) >= 3:
                critical.append(node)
        return critical

    def _find_bottlenecks(self) -> List[CausalNode]:
        """Find bottleneck nodes (many dependencies converge)"""
        bottlenecks = []
        for node_id, node in self.graph.nodes.items():
            # Bottleneck if has many incoming edges
            if len(self.graph.incoming[node_id]) >= 3:
                bottlenecks.append(node)
        return bottlenecks

    def _find_assertions(self, tree: ast.AST) -> List[ast.Assert]:
        """Find assertion statements"""
        assertions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assertions.append(node)
        return assertions

    def _extract_vars_from_assertion(self, assertion: ast.Assert) -> Set[str]:
        """Extract variables from assertion"""
        return self._extract_dependencies(assertion.test)

    def _check_causal_risk(self, variables: Set[str], graph: CausalGraph) -> bool:
        """Check if variables have risky causal paths"""
        for var in variables:
            # Find node for variable
            var_node_id = None
            for node_id, node in graph.nodes.items():
                if node.name == var:
                    var_node_id = node_id
                    break

            if not var_node_id:
                continue

            # Check for cycles (potential infinite loop)
            if any(var_node_id in [n.node_id for n in cycle]
                  for cycle in graph.detect_cycles()):
                return True

            # Check for weak causal paths
            # (Would need more sophisticated analysis in production)

        return False

    def _explain_causal_path(self, path: List[CausalNode]) -> str:
        """Generate explanation of causal path"""
        if len(path) <= 1:
            return "Direct cause"

        steps = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            steps.append(f"{source.name} → {target.name}")

        return " → ".join(steps)


# Example usage
def example_usage():
    """Example of causal analysis"""
    analyzer = CausalAnalyzer()

    # Example code
    source_code = """
def process_data(input_data):
    # Step 1: Validate
    if not input_data:
        raise ValueError("Input required")

    # Step 2: Transform
    cleaned = clean(input_data)
    normalized = normalize(cleaned)

    # Step 3: Compute
    result = compute(normalized)

    return result
"""

    # Analyze
    result = analyzer.analyze_code(source_code)

    # Print results
    print("\n=== Root Causes ===")
    for node, distance in result.root_causes:
        print(f"  {node.name} (distance: {distance})")

    print("\n=== Critical Nodes ===")
    for node in result.critical_nodes:
        print(f"  {node.name} (affects {len(result.graph.get_effects(node.node_id))} nodes)")

    print("\n=== Cycles ===")
    for cycle in result.cycles:
        print(f"  {' → '.join(n.name for n in cycle)}")

    # Test prediction example
    test_code = """
def test_process():
    data = get_test_data()
    result = process_data(data)
    assert result is not None
"""

    prediction, confidence = analyzer.predict_test_outcome(test_code, result)
    print(f"\n=== Test Prediction ===")
    print(f"  Prediction: {prediction}")
    print(f"  Confidence: {confidence:.0%}")


if __name__ == "__main__":
    example_usage()
