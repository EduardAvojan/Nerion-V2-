"""Action definitions and AST transformers for the Nerion environment."""

from __future__ import annotations

import ast
from enum import Enum, auto


class Action(Enum):
    """Supported environment actions."""

    RENAME_LOCAL_VARIABLE_IN_ADD = auto()
    CHANGE_OPERATOR_MULTIPLY_TO_ADD = auto()
    IMPLEMENT_MULTIPLY_DOCSTRING = auto()


class StatefulRenameVisitor(ast.NodeVisitor):
    """Collect nodes to rename within a specific function scope."""

    def __init__(self, target_function: str, old_name: str, new_name: str):
        self.target_function = target_function
        self.old_name = old_name
        self.new_name = new_name
        self._current_function: str | None = None
        self.nodes_to_rename: list[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - simple traversal
        previous_function = self._current_function
        self._current_function = node.name
        self.generic_visit(node)
        self._current_function = previous_function

    def visit_arg(self, node: ast.arg) -> None:
        if self._current_function == self.target_function and node.arg == self.old_name:
            self.nodes_to_rename.append(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if self._current_function == self.target_function and node.id == self.old_name:
            self.nodes_to_rename.append(node)
        self.generic_visit(node)


__all__ = ["Action", "StatefulRenameVisitor"]
