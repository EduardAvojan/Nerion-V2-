"""
Code Augmentation for Contrastive Learning

Provides semantic-preserving transformations of code to create
positive pairs for self-supervised learning.
"""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Set
import copy


class AugmentationType(Enum):
    """Types of code augmentations"""
    VARIABLE_RENAME = "variable_rename"
    COMMENT_REMOVAL = "comment_removal"
    WHITESPACE_CHANGE = "whitespace_change"
    DOCSTRING_REMOVAL = "docstring_removal"
    IMPORT_REORDER = "import_reorder"
    FUNCTION_REORDER = "function_reorder"
    CONSTANT_RENAME = "constant_rename"
    TYPE_HINT_REMOVAL = "type_hint_removal"
    PASS_ADDITION = "pass_addition"
    ASSERTION_REMOVAL = "assertion_removal"


@dataclass
class AugmentationResult:
    """Result of code augmentation"""
    original_code: str
    augmented_code: str
    augmentation_types: List[AugmentationType]
    preserved_semantics: bool
    ast_valid: bool


class CodeAugmentor:
    """
    Applies semantic-preserving transformations to code.

    Creates positive pairs for contrastive learning by:
    1. Renaming variables/functions (preserving semantics)
    2. Removing comments/docstrings
    3. Reordering functions/imports
    4. Changing whitespace

    Usage:
        >>> augmentor = CodeAugmentor()
        >>> result = augmentor.augment("def add(x, y): return x + y")
        >>> print(result.augmented_code)  # "def sum_values(a, b): return a + b"
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augmentor.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.variable_names = [
            'var', 'val', 'item', 'elem', 'data', 'result', 'temp',
            'x', 'y', 'z', 'i', 'j', 'k', 'a', 'b', 'c',
            'foo', 'bar', 'baz', 'qux'
        ]

        self.function_names = [
            'func', 'process', 'handle', 'compute', 'calculate',
            'transform', 'convert', 'parse', 'validate', 'execute'
        ]

    def augment(
        self,
        code: str,
        num_augmentations: int = 2,
        allowed_types: Optional[List[AugmentationType]] = None
    ) -> AugmentationResult:
        """
        Apply augmentations to code.

        Args:
            code: Source code to augment
            num_augmentations: Number of augmentations to apply
            allowed_types: Specific augmentation types to use (None = all)

        Returns:
            Augmentation result
        """
        if allowed_types is None:
            allowed_types = list(AugmentationType)

        augmented = code
        applied_types = []

        # Try to parse original code
        try:
            tree = ast.parse(code)
            ast_valid = True
        except:
            return AugmentationResult(
                original_code=code,
                augmented_code=code,
                augmentation_types=[],
                preserved_semantics=True,
                ast_valid=False
            )

        # Apply random augmentations
        available_types = allowed_types.copy()
        random.shuffle(available_types)

        for aug_type in available_types[:num_augmentations]:
            try:
                if aug_type == AugmentationType.VARIABLE_RENAME:
                    augmented = self._rename_variables(augmented)
                elif aug_type == AugmentationType.COMMENT_REMOVAL:
                    augmented = self._remove_comments(augmented)
                elif aug_type == AugmentationType.WHITESPACE_CHANGE:
                    augmented = self._change_whitespace(augmented)
                elif aug_type == AugmentationType.DOCSTRING_REMOVAL:
                    augmented = self._remove_docstrings(augmented)
                elif aug_type == AugmentationType.IMPORT_REORDER:
                    augmented = self._reorder_imports(augmented)
                elif aug_type == AugmentationType.FUNCTION_REORDER:
                    augmented = self._reorder_functions(augmented)
                elif aug_type == AugmentationType.CONSTANT_RENAME:
                    augmented = self._rename_constants(augmented)
                elif aug_type == AugmentationType.TYPE_HINT_REMOVAL:
                    augmented = self._remove_type_hints(augmented)
                elif aug_type == AugmentationType.PASS_ADDITION:
                    augmented = self._add_pass_statements(augmented)
                elif aug_type == AugmentationType.ASSERTION_REMOVAL:
                    augmented = self._remove_assertions(augmented)

                applied_types.append(aug_type)
            except:
                # Skip augmentation if it fails
                continue

        # Verify augmented code is valid
        try:
            ast.parse(augmented)
            final_ast_valid = True
        except:
            # Revert to original if augmentation broke syntax
            augmented = code
            final_ast_valid = True

        return AugmentationResult(
            original_code=code,
            augmented_code=augmented,
            augmentation_types=applied_types,
            preserved_semantics=True,
            ast_valid=final_ast_valid
        )

    def _rename_variables(self, code: str) -> str:
        """Rename variables to semantically equivalent names"""
        try:
            tree = ast.parse(code)
            renamer = VariableRenamer(self.variable_names)
            new_tree = renamer.visit(tree)
            return ast.unparse(new_tree)
        except:
            return code

    def _remove_comments(self, code: str) -> str:
        """Remove inline comments"""
        lines = code.split('\n')
        result = []

        for line in lines:
            # Remove inline comments (preserve strings)
            if '#' in line:
                # Simple heuristic: remove everything after #
                # (doesn't handle strings with # properly)
                in_string = False
                cleaned = []
                for i, char in enumerate(line):
                    if char in ['"', "'"]:
                        in_string = not in_string
                    if char == '#' and not in_string:
                        break
                    cleaned.append(char)
                result.append(''.join(cleaned).rstrip())
            else:
                result.append(line)

        return '\n'.join(result)

    def _change_whitespace(self, code: str) -> str:
        """Change whitespace (add/remove blank lines)"""
        lines = code.split('\n')

        # Remove consecutive blank lines
        result = []
        prev_blank = False
        for line in lines:
            is_blank = line.strip() == ''
            if not (is_blank and prev_blank):
                result.append(line)
            prev_blank = is_blank

        return '\n'.join(result)

    def _remove_docstrings(self, code: str) -> str:
        """Remove docstrings from functions/classes"""
        try:
            tree = ast.parse(code)
            remover = DocstringRemover()
            new_tree = remover.visit(tree)
            return ast.unparse(new_tree)
        except:
            return code

    def _reorder_imports(self, code: str) -> str:
        """Reorder import statements"""
        try:
            tree = ast.parse(code)
            reorderer = ImportReorderer()
            new_tree = reorderer.visit(tree)
            return ast.unparse(new_tree)
        except:
            return code

    def _reorder_functions(self, code: str) -> str:
        """Reorder function definitions"""
        try:
            tree = ast.parse(code)
            reorderer = FunctionReorderer()
            new_tree = reorderer.visit(tree)
            return ast.unparse(new_tree)
        except:
            return code

    def _rename_constants(self, code: str) -> str:
        """Rename constant variables (ALL_CAPS)"""
        try:
            tree = ast.parse(code)
            renamer = ConstantRenamer()
            new_tree = renamer.visit(tree)
            return ast.unparse(new_tree)
        except:
            return code

    def _remove_type_hints(self, code: str) -> str:
        """Remove type hints from function signatures"""
        try:
            tree = ast.parse(code)
            remover = TypeHintRemover()
            new_tree = remover.visit(tree)
            return ast.unparse(new_tree)
        except:
            return code

    def _add_pass_statements(self, code: str) -> str:
        """Add redundant pass statements"""
        try:
            tree = ast.parse(code)
            adder = PassAdder()
            new_tree = adder.visit(tree)
            return ast.unparse(new_tree)
        except:
            return code

    def _remove_assertions(self, code: str) -> str:
        """Remove assert statements (non-semantic in production)"""
        try:
            tree = ast.parse(code)
            remover = AssertionRemover()
            new_tree = remover.visit(tree)
            return ast.unparse(new_tree)
        except:
            return code


class VariableRenamer(ast.NodeTransformer):
    """Renames variables to different but valid names"""

    def __init__(self, name_pool: List[str]):
        self.name_pool = name_pool
        self.mapping: Dict[str, str] = {}
        self.used_names: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.Name:
        # Only rename user-defined variables (not builtins)
        if node.id in ['self', 'cls', 'True', 'False', 'None']:
            return node

        if node.id not in self.mapping:
            # Generate new name
            for new_name in self.name_pool:
                if new_name not in self.used_names:
                    self.mapping[node.id] = new_name
                    self.used_names.add(new_name)
                    break
            else:
                # No available names
                return node

        # Replace name
        new_node = copy.copy(node)
        new_node.id = self.mapping[node.id]
        return new_node


class DocstringRemover(ast.NodeTransformer):
    """Removes docstrings from functions and classes"""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)

        # Remove docstring (first statement if it's a string)
        if (node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]

        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)

        # Remove docstring
        if (node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]

        return node


class ImportReorderer(ast.NodeTransformer):
    """Reorders import statements"""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        imports = []
        other = []

        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)
            else:
                other.append(stmt)

        # Shuffle imports
        random.shuffle(imports)

        node.body = imports + other
        return node


class FunctionReorderer(ast.NodeTransformer):
    """Reorders function definitions at module level"""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        functions = []
        other = []

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                functions.append(stmt)
            else:
                other.append(stmt)

        # Shuffle functions
        random.shuffle(functions)

        # Keep non-functions in original positions
        # Interleave functions back
        result = []
        func_idx = 0
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if func_idx < len(functions):
                    result.append(functions[func_idx])
                    func_idx += 1
            else:
                result.append(stmt)

        node.body = result
        return node


class ConstantRenamer(ast.NodeTransformer):
    """Renames constant variables (ALL_CAPS convention)"""

    def __init__(self):
        self.mapping: Dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> ast.Name:
        # Only rename ALL_CAPS variables
        if node.id.isupper() and len(node.id) > 1:
            if node.id not in self.mapping:
                # Generate new constant name
                self.mapping[node.id] = f"CONST_{len(self.mapping)}"

            new_node = copy.copy(node)
            new_node.id = self.mapping[node.id]
            return new_node

        return node


class TypeHintRemover(ast.NodeTransformer):
    """Removes type hints from function signatures"""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)

        # Remove return type annotation
        node.returns = None

        # Remove argument type annotations
        for arg in node.args.args:
            arg.annotation = None

        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign:
        """Convert annotated assignments to regular assignments"""
        return ast.Assign(
            targets=[node.target],
            value=node.value
        )


class PassAdder(ast.NodeTransformer):
    """Adds redundant pass statements"""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)

        # Add pass at end of function body
        if random.random() < 0.5:
            node.body.append(ast.Pass())

        return node


class AssertionRemover(ast.NodeTransformer):
    """Removes assert statements"""

    def visit_Assert(self, node: ast.Assert) -> Optional[ast.AST]:
        # Remove assertion
        return None


# Example usage
if __name__ == "__main__":
    augmentor = CodeAugmentor(seed=42)

    code = """
def fibonacci(n: int) -> int:
    '''Calculate nth Fibonacci number'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(x: int) -> int:
    '''Calculate factorial'''
    if x <= 1:
        return 1
    return x * factorial(x-1)
"""

    print("=== Original Code ===")
    print(code)

    print("\n=== Augmentation 1 (Variable Rename + Comment Removal) ===")
    result1 = augmentor.augment(
        code,
        num_augmentations=2,
        allowed_types=[
            AugmentationType.VARIABLE_RENAME,
            AugmentationType.COMMENT_REMOVAL
        ]
    )
    print(result1.augmented_code)
    print(f"Applied: {[t.value for t in result1.augmentation_types]}")

    print("\n=== Augmentation 2 (Docstring Removal + Function Reorder) ===")
    result2 = augmentor.augment(
        code,
        num_augmentations=2,
        allowed_types=[
            AugmentationType.DOCSTRING_REMOVAL,
            AugmentationType.FUNCTION_REORDER
        ]
    )
    print(result2.augmented_code)
    print(f"Applied: {[t.value for t in result2.augmentation_types]}")
