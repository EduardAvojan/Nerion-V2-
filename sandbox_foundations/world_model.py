"""Phase 1 World Model: networkx graph extraction from the AST."""

import ast
import os

import networkx as nx

from scaffold import build_ast_from_file


class AstGraphVisitor(ast.NodeVisitor):
    """AST visitor that builds a directed graph of function definitions."""

    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Add each function definition as a node in the graph."""
        self.graph.add_node(node.name, node_type="function")

        # Track the function context while visiting children
        previous_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = previous_function


def main():
    """Build and display the toy universe world model graph."""
    file_path = os.path.join(os.path.dirname(__file__), "math_logic.py")
    print(f"Building World Model for: {file_path}\n")

    code_ast = build_ast_from_file(file_path)

    if code_ast:
        visitor = AstGraphVisitor()
        visitor.visit(code_ast)
        world_model_graph = visitor.graph

        print("--- World Model Graph ---")
        print(f"Nodes: {list(world_model_graph.nodes(data=True))}")
        print(f"Edges: {list(world_model_graph.edges())}")
        print("\n--- End World Model Graph ---")


if __name__ == "__main__":
    main()
