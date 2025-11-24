"""
Symbolic Executor

Executes code symbolically to explore multiple execution paths.
Tracks constraints and variable states without concrete values.
"""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple


@dataclass
class SymbolicValue:
    """Represents a symbolic value (variable without concrete value)"""
    name: str
    type_hint: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    possible_values: Set[Any] = field(default_factory=set)

    def __repr__(self):
        return f"Sym({self.name}, type={self.type_hint})"


@dataclass
class ExecutionPath:
    """Represents one possible execution path"""
    path_id: int
    conditions: List[str] = field(default_factory=list)
    variables: Dict[str, SymbolicValue] = field(default_factory=dict)
    return_value: Optional[SymbolicValue] = None
    feasible: bool = True
    error_type: Optional[str] = None


@dataclass
class SymbolicExecutionResult:
    """Result of symbolic execution"""
    paths: List[ExecutionPath]
    return_value: Optional[SymbolicValue]
    modified_variables: Set[str]
    functions_called: List[str]
    io_operations: List[str]
    complete: bool  # True if all paths fully explored
    error_paths: List[ExecutionPath] = field(default_factory=list)


class SymbolicExecutor:
    """
    Symbolic execution engine.

    Explores all possible execution paths by treating variables
    as symbolic rather than concrete values.

    Usage:
        >>> executor = SymbolicExecutor()
        >>> code = "if x > 0: y = 1 else: y = -1"
        >>> result = executor.execute(ast.parse(code), {'x': SymbolicValue('x')})
        >>> len(result.paths)  # 2 paths (x > 0 and x <= 0)
    """

    def __init__(self):
        self.max_paths = 1000
        self.max_depth = 100

    def execute(
        self,
        tree: ast.AST,
        initial_state: Dict[str, Any],
        max_paths: int = 1000
    ) -> SymbolicExecutionResult:
        """
        Symbolically execute AST.

        Args:
            tree: AST to execute
            initial_state: Initial variable state
            max_paths: Maximum paths to explore

        Returns:
            Execution result with all paths
        """
        self.max_paths = max_paths

        # Convert initial state to symbolic values
        symbolic_state = {}
        for var, value in initial_state.items():
            if isinstance(value, SymbolicValue):
                symbolic_state[var] = value
            else:
                symbolic_state[var] = SymbolicValue(
                    name=var,
                    type_hint=type(value).__name__,
                    possible_values={value}
                )

        # Initialize paths
        paths = [ExecutionPath(
            path_id=0,
            variables=symbolic_state.copy()
        )]

        # Track metadata
        modified_variables = set()
        functions_called = []
        io_operations = []
        error_paths = []

        # Execute each statement
        try:
            for node in tree.body:
                new_paths = []

                for path in paths:
                    if not path.feasible:
                        new_paths.append(path)
                        continue

                    # Execute node on this path
                    result_paths = self._execute_node(
                        node,
                        path,
                        functions_called,
                        io_operations,
                        modified_variables
                    )

                    new_paths.extend(result_paths)

                    # Check path limit
                    if len(new_paths) > max_paths:
                        break

                paths = new_paths[:max_paths]

        except Exception as e:
            # Execution error - mark all paths as errored
            for path in paths:
                path.feasible = False
                path.error_type = str(e)
                error_paths.append(path)

        # Extract return values
        return_value = None
        for path in paths:
            if path.return_value:
                return_value = path.return_value
                break

        # Check if execution was complete
        complete = len(paths) < max_paths

        return SymbolicExecutionResult(
            paths=paths,
            return_value=return_value,
            modified_variables=modified_variables,
            functions_called=functions_called,
            io_operations=io_operations,
            complete=complete,
            error_paths=error_paths
        )

    def _execute_node(
        self,
        node: ast.AST,
        path: ExecutionPath,
        functions_called: List[str],
        io_operations: List[str],
        modified_variables: Set[str]
    ) -> List[ExecutionPath]:
        """Execute a single AST node on a path"""

        # Assignment
        if isinstance(node, ast.Assign):
            return self._execute_assign(node, path, modified_variables)

        # If statement (branches into multiple paths)
        elif isinstance(node, ast.If):
            return self._execute_if(node, path, functions_called, io_operations, modified_variables)

        # Function definition
        elif isinstance(node, ast.FunctionDef):
            return [path]  # Just record, don't execute

        # Return statement
        elif isinstance(node, ast.Return):
            return self._execute_return(node, path)

        # Expression statement
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                self._record_call(node.value, functions_called, io_operations)
            return [path]

        # For/While loops (simplified - execute once)
        elif isinstance(node, (ast.For, ast.While)):
            return self._execute_loop(node, path, functions_called, io_operations, modified_variables)

        else:
            # Unknown node type - just continue
            return [path]

    def _execute_assign(
        self,
        node: ast.Assign,
        path: ExecutionPath,
        modified_variables: Set[str]
    ) -> List[ExecutionPath]:
        """Execute assignment"""
        # Extract target variable name
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # Create symbolic value for assignment
                value = self._eval_expr(node.value, path)

                path.variables[var_name] = value
                modified_variables.add(var_name)

        return [path]

    def _execute_if(
        self,
        node: ast.If,
        path: ExecutionPath,
        functions_called: List[str],
        io_operations: List[str],
        modified_variables: Set[str]
    ) -> List[ExecutionPath]:
        """Execute if statement - branches into two paths"""
        # Get condition
        condition = ast.unparse(node.test)

        # Create two paths: one for true branch, one for false
        true_path = ExecutionPath(
            path_id=len(path.conditions),
            conditions=path.conditions + [condition],
            variables=path.variables.copy(),
            feasible=True
        )

        false_path = ExecutionPath(
            path_id=len(path.conditions) + 1,
            conditions=path.conditions + [f"not ({condition})"],
            variables=path.variables.copy(),
            feasible=True
        )

        paths = []

        # Execute true branch
        for stmt in node.body:
            true_paths = self._execute_node(stmt, true_path, functions_called, io_operations, modified_variables)
            if true_paths:
                true_path = true_paths[0]  # Simplified: take first path

        paths.append(true_path)

        # Execute false branch (else/elif)
        if node.orelse:
            for stmt in node.orelse:
                false_paths = self._execute_node(stmt, false_path, functions_called, io_operations, modified_variables)
                if false_paths:
                    false_path = false_paths[0]

            paths.append(false_path)
        else:
            paths.append(false_path)

        return paths

    def _execute_loop(
        self,
        node: ast.AST,
        path: ExecutionPath,
        functions_called: List[str],
        io_operations: List[str],
        modified_variables: Set[str]
    ) -> List[ExecutionPath]:
        """Execute loop (simplified - just execute body once)"""
        # For symbolic execution, we can't execute loops fully
        # Just execute body once to capture effects

        for stmt in node.body:
            paths = self._execute_node(stmt, path, functions_called, io_operations, modified_variables)
            if paths:
                path = paths[0]

        return [path]

    def _execute_return(
        self,
        node: ast.Return,
        path: ExecutionPath
    ) -> List[ExecutionPath]:
        """Execute return statement"""
        if node.value:
            path.return_value = self._eval_expr(node.value, path)

        return [path]

    def _eval_expr(
        self,
        expr: ast.expr,
        path: ExecutionPath
    ) -> SymbolicValue:
        """Evaluate expression symbolically"""

        # Constant
        if isinstance(expr, ast.Constant):
            return SymbolicValue(
                name=str(expr.value),
                type_hint=type(expr.value).__name__,
                possible_values={expr.value}
            )

        # Variable reference
        elif isinstance(expr, ast.Name):
            if expr.id in path.variables:
                return path.variables[expr.id]
            else:
                # Unknown variable
                return SymbolicValue(name=expr.id)

        # Binary operation
        elif isinstance(expr, ast.BinOp):
            left = self._eval_expr(expr.left, path)
            right = self._eval_expr(expr.right, path)

            # Create symbolic result
            op_name = expr.op.__class__.__name__
            return SymbolicValue(
                name=f"({left.name} {op_name} {right.name})",
                constraints=[f"result of {op_name}"]
            )

        # Function call
        elif isinstance(expr, ast.Call):
            func_name = "unknown"
            if isinstance(expr.func, ast.Name):
                func_name = expr.func.id

            return SymbolicValue(
                name=f"call_{func_name}",
                constraints=[f"return value of {func_name}"]
            )

        # Default: unknown
        else:
            return SymbolicValue(name="unknown")

    def _record_call(
        self,
        call: ast.Call,
        functions_called: List[str],
        io_operations: List[str]
    ):
        """Record function call"""
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
            functions_called.append(func_name)

            # Check for I/O operations
            if func_name in ['open', 'read', 'write', 'print']:
                io_operations.append(func_name)

        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
            functions_called.append(func_name)


# Example usage
if __name__ == "__main__":
    executor = SymbolicExecutor()

    # Test 1: Simple branching
    code = """
if x > 0:
    y = 1
else:
    y = -1
"""

    tree = ast.parse(code)
    result = executor.execute(tree, {'x': SymbolicValue('x', type_hint='int')})

    print(f"Test 1 - Paths explored: {len(result.paths)}")
    for i, path in enumerate(result.paths):
        print(f"  Path {i}: conditions={path.conditions}, feasible={path.feasible}")

    # Test 2: Function with return
    code2 = """
def max_val(a, b):
    if a > b:
        return a
    else:
        return b
"""

    tree2 = ast.parse(code2)
    result2 = executor.execute(tree2, {
        'a': SymbolicValue('a', type_hint='int'),
        'b': SymbolicValue('b', type_hint='int')
    })

    print(f"\nTest 2 - Paths explored: {len(result2.paths)}")
    print(f"  Return value: {result2.return_value}")
