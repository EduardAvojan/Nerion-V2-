"""Multi-language code parser using tree-sitter for Python, TypeScript, JavaScript, Go, Rust, and Java."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import tree_sitter_python as ts_python
import tree_sitter_javascript as ts_javascript
import tree_sitter_typescript as ts_typescript
import tree_sitter_go as ts_go
import tree_sitter_rust as ts_rust
import tree_sitter_java as ts_java
from tree_sitter import Language, Parser, Node


class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"


# Language instances
LANGUAGES: Dict[CodeLanguage, Language] = {
    CodeLanguage.PYTHON: Language(ts_python.language()),
    CodeLanguage.JAVASCRIPT: Language(ts_javascript.language()),
    CodeLanguage.TYPESCRIPT: Language(ts_typescript.language_typescript()),
    CodeLanguage.GO: Language(ts_go.language()),
    CodeLanguage.RUST: Language(ts_rust.language()),
    CodeLanguage.JAVA: Language(ts_java.language()),
}


@dataclass
class FunctionInfo:
    """Extracted information about a function."""
    name: str
    line_number: int
    line_count: int
    arg_count: int
    avg_arg_length: float
    docstring_length: int
    branch_count: int
    call_count: int
    return_count: int
    cyclomatic_complexity: int
    call_targets: List[str]
    reads: Set[str]
    writes: Set[str]


@dataclass
class StatementInfo:
    """Extracted information about a statement."""
    name: str
    line_number: int
    line_count: int
    nested_complexity: int
    reads: Set[str]
    writes: Set[str]


@dataclass
class ExpressionInfo:
    """Extracted information about an expression."""
    name: str
    line_number: int
    operator_type: str
    arg_count: int


def detect_language(source_code: str) -> CodeLanguage:
    """Detect the programming language from source code."""
    # Simple heuristics based on syntax patterns
    # Check first 1000 chars for language-specific patterns (increased from 500)
    code_start = source_code[:1000].strip()

    # Rust indicators (check EARLY - unique syntax)
    if any(pattern in code_start for pattern in ['fn ', 'impl ', 'pub fn', 'let mut', 'use std::', 'use crate::', 'fn main(', '// run-rustfix']):
        return CodeLanguage.RUST

    # Java indicators (check early - very distinct)
    if any(pattern in code_start for pattern in ['public class', 'private class', 'protected class', 'import java.', 'public static void main']):
        return CodeLanguage.JAVA

    # TypeScript/JavaScript (check BEFORE Go - they share 'import' and 'type')
    # Strong TypeScript markers
    if any(pattern in code_start for pattern in ['interface ', ': string', ': number', 'export interface', '<T>', ': boolean', 'as const', 'typeof ', 'keyof ']):
        return CodeLanguage.TYPESCRIPT

    # Check for ES6 import/export syntax (TypeScript/JavaScript)
    has_import_from = 'import ' in code_start and ' from ' in code_start
    has_export = 'export ' in code_start

    # JavaScript/TypeScript indicators
    if any(pattern in code_start for pattern in ['const ', 'let ', 'var ', 'function ', 'require(', '=>', '$j.fn.', '$.fn.', 'jQuery']):
        # If has type annotations, it's TypeScript
        if has_import_from or has_export or 'interface ' in code_start or (': ' in code_start and any(t in code_start for t in [': string', ': number', ': boolean'])):
            return CodeLanguage.TYPESCRIPT
        return CodeLanguage.JAVASCRIPT

    # Check for import from syntax (strongly suggests TypeScript/JavaScript, not Go)
    if has_import_from:
        # Double check if it's TypeScript
        if any(pattern in code_start for pattern in ['interface ', ': ', 'type ', 'as ', 'typeof ']):
            return CodeLanguage.TYPESCRIPT
        return CodeLanguage.JAVASCRIPT

    # Go indicators (check AFTER TypeScript/JavaScript)
    if any(pattern in code_start for pattern in ['package ', 'func ', 'import (', 'func main(']):
        return CodeLanguage.GO

    # Python indicators (check LAST since it shares many keywords)
    if any(pattern in code_start for pattern in ['def ', 'import ', 'from ', '__init__', 'self.', 'async def', 'await ', 'print(']):
        return CodeLanguage.PYTHON

    # If has 'class' but no other language markers, likely Python
    if 'class ' in code_start:
        return CodeLanguage.PYTHON

    # Default to Python
    return CodeLanguage.PYTHON


def parse_source(source_code: str, language: Optional[CodeLanguage] = None) -> Tuple[Node, CodeLanguage]:
    """Parse source code using tree-sitter."""
    if language is None:
        language = detect_language(source_code)

    parser = Parser(LANGUAGES[language])
    tree = parser.parse(bytes(source_code, "utf-8"))
    return tree.root_node, language


def _count_branches_ts(node: Node) -> int:
    """Count branching constructs in a tree-sitter node."""
    branch_types = {
        'if_statement', 'for_statement', 'while_statement', 'try_statement',
        'switch_statement', 'match_expression', 'conditional_expression',
        # Go
        'for_clause', 'if_statement', 'switch_statement',
        # Rust
        'if_expression', 'match_expression', 'for_expression', 'while_expression',
        # Java
        'if_statement', 'for_statement', 'while_statement', 'switch_expression',
    }

    count = 0

    def traverse(n: Node) -> None:
        nonlocal count
        if n.type in branch_types:
            count += 1
        for child in n.children:
            traverse(child)

    traverse(node)
    return count


def _count_returns_ts(node: Node) -> int:
    """Count return statements in a tree-sitter node."""
    return_types = {'return_statement', 'return_expression'}

    count = 0

    def traverse(n: Node) -> None:
        nonlocal count
        if n.type in return_types:
            count += 1
        for child in n.children:
            traverse(child)

    traverse(node)
    return count


def _count_calls_ts(node: Node) -> int:
    """Count function calls in a tree-sitter node."""
    call_types = {'call_expression', 'call', 'method_invocation'}

    count = 0

    def traverse(n: Node) -> None:
        nonlocal count
        if n.type in call_types:
            count += 1
        for child in n.children:
            traverse(child)

    traverse(node)
    return count


def _get_call_targets_ts(node: Node, source_bytes: bytes) -> List[str]:
    """Extract function call targets from a tree-sitter node."""
    call_types = {'call_expression', 'call', 'method_invocation'}
    targets = []

    def traverse(n: Node) -> None:
        if n.type in call_types:
            # Try to get the function name
            if n.children:
                func_node = n.children[0]
                if func_node.type in {'identifier', 'field_identifier'}:
                    name = source_bytes[func_node.start_byte:func_node.end_byte].decode('utf-8')
                    targets.append(name)
        for child in n.children:
            traverse(child)

    traverse(node)
    return targets


def _get_identifiers_ts(node: Node, source_bytes: bytes, read_contexts: Set[str], write_contexts: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Extract variable reads and writes from a tree-sitter node."""
    reads: Set[str] = set()
    writes: Set[str] = set()

    def traverse(n: Node, is_write: bool = False) -> None:
        # Check if this node is in a write context
        if n.type in write_contexts:
            is_write = True
        elif n.type in read_contexts:
            is_write = False

        if n.type == 'identifier':
            name = source_bytes[n.start_byte:n.end_byte].decode('utf-8')
            if is_write:
                writes.add(name)
            else:
                reads.add(name)

        for child in n.children:
            traverse(child, is_write)

    traverse(node)
    return reads, writes


def extract_functions_ts(root: Node, source_code: str, language: CodeLanguage) -> List[FunctionInfo]:
    """Extract ALL callable units from a tree-sitter parse tree (functions, methods, constructors)."""
    functions = []
    source_bytes = bytes(source_code, "utf-8")

    # Language-specific callable node types - COMPREHENSIVE (ALL patterns)
    func_types_map = {
        CodeLanguage.PYTHON: {
            'function_definition',  # def foo(), async def, nested functions, all methods
            'lambda',               # lambda x: x * 2
        },
        CodeLanguage.JAVASCRIPT: {
            'function_declaration',  # function foo() {}
            'function',              # anonymous function
            'arrow_function',        # () => {}
            'method_definition',     # class methods (including static, async, generator)
            'function_expression',   # const f = function() {}
            'accessor_declaration',  # get/set property accessors
        },
        CodeLanguage.TYPESCRIPT: {
            'function_declaration',  # function foo() {}
            'function',              # anonymous function
            'arrow_function',        # () => {}
            'method_definition',     # class methods (including static, async, generator)
            'method_signature',      # interface method signatures
            'function_signature',    # function type declarations
            'accessor_declaration',  # get/set property accessors
        },
        CodeLanguage.GO: {
            'function_declaration',  # func foo() {}, func literals
            'method_declaration',    # func (r *Receiver) foo() {}
        },
        CodeLanguage.RUST: {
            'function_item',         # fn foo() {}, async fn, const fn, unsafe fn
            'function_signature_item',  # trait function signatures
            'closure_expression',    # |x| x + 1
        },
        CodeLanguage.JAVA: {
            'method_declaration',    # public void foo() {}, static methods, abstract
            'constructor_declaration',  # public MyClass() {}
            'lambda_expression',     # (x) -> x * 2
        },
    }

    func_types = func_types_map.get(language, set())

    # Language-specific identifier contexts
    write_contexts_map = {
        CodeLanguage.PYTHON: {'assignment', 'augmented_assignment'},
        CodeLanguage.JAVASCRIPT: {'variable_declarator', 'assignment_expression'},
        CodeLanguage.TYPESCRIPT: {'variable_declarator', 'assignment_expression'},
        CodeLanguage.GO: {'short_var_declaration', 'assignment_statement'},
        CodeLanguage.RUST: {'let_declaration', 'assignment_expression'},
        CodeLanguage.JAVA: {'variable_declarator', 'assignment_expression'},
    }

    read_contexts = {'expression', 'call_expression', 'binary_expression'}
    write_contexts = write_contexts_map.get(language, set())

    # Helper to find parent class name
    def get_parent_class_name(node: Node) -> Optional[str]:
        """Walk up the tree to find if this function is inside a class."""
        current = node.parent
        while current:
            if current.type in {'class_definition', 'class_declaration', 'class_body', 'interface_declaration'}:
                # Try to find the class name
                for child in current.children:
                    if child.type == 'identifier' or child.type == 'type_identifier':
                        return source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                return None
            current = current.parent
        return None

    def traverse(node: Node) -> None:
        if node.type in func_types:
            # Extract function/method name
            func_name = "unknown"

            # Special handling for anonymous/lambda functions
            if node.type in {'lambda', 'lambda_expression', 'closure_expression', 'arrow_function'}:
                # Use line number as identifier for anonymous functions
                func_name = f"lambda_{node.start_point[0] + 1}"
            elif node.type == 'accessor_declaration':
                # For getters/setters, extract property name
                for child in node.children:
                    if child.type in {'identifier', 'property_identifier'}:
                        func_name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                        # Check if it's a getter or setter
                        if any(c.type == 'get' for c in node.children):
                            func_name = f"get_{func_name}"
                        elif any(c.type == 'set' for c in node.children):
                            func_name = f"set_{func_name}"
                        break
            elif node.type in {'function', 'function_expression'}:
                # Anonymous function expression - check if assigned to something
                # Look at parent for assignment context
                parent = node.parent
                if parent:
                    # Check for patterns like: obj.prop = function() {} or var x = function() {}
                    if parent.type == 'assignment_expression':
                        # Look for left side of assignment
                        if parent.children:
                            left_side = parent.children[0]
                            if left_side.type == 'member_expression':
                                # Extract property name (e.g., $.fn.arrowSteps)
                                func_name = source_bytes[left_side.start_byte:left_side.end_byte].decode('utf-8')
                            elif left_side.type == 'identifier':
                                func_name = source_bytes[left_side.start_byte:left_side.end_byte].decode('utf-8')
                    elif parent.type == 'variable_declarator':
                        # var x = function() {}
                        for sibling in parent.children:
                            if sibling.type in {'identifier', 'property_identifier'}:
                                func_name = source_bytes[sibling.start_byte:sibling.end_byte].decode('utf-8')
                                break

                # If still unknown, use line number
                if func_name == "unknown":
                    func_name = f"anonymous_{node.start_point[0] + 1}"
            else:
                # Regular function/method name extraction
                for child in node.children:
                    if child.type in {'identifier', 'property_identifier'}:
                        func_name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
                        break

            # If inside a class, prefix with class name (skip for top-level lambdas)
            if node.type not in {'lambda', 'lambda_expression', 'closure_expression'}:
                class_name = get_parent_class_name(node)
                if class_name:
                    func_name = f"{class_name}.{func_name}"

            # Get line numbers
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            line_count = end_line - start_line + 1

            # Count parameters
            param_count = 0
            total_param_length = 0

            for child in node.children:
                if child.type in {'parameters', 'parameter_list', 'formal_parameters'}:
                    params = [c for c in child.children if c.type in {'identifier', 'parameter_declaration', 'formal_parameter'}]
                    param_count = len(params)
                    for param in params:
                        param_text = source_bytes[param.start_byte:param.end_byte].decode('utf-8')
                        total_param_length += len(param_text)

            avg_param_length = (total_param_length / param_count) if param_count > 0 else 0.0

            # Extract docstring (only for Python)
            docstring_length = 0
            if language == CodeLanguage.PYTHON:
                body = next((c for c in node.children if c.type == 'block'), None)
                if body and body.children:
                    first_stmt = body.children[0]
                    if first_stmt.type == 'expression_statement':
                        expr = first_stmt.children[0] if first_stmt.children else None
                        if expr and expr.type == 'string':
                            docstring_length = expr.end_byte - expr.start_byte

            # Count metrics
            branch_count = _count_branches_ts(node)
            call_count = _count_calls_ts(node)
            return_count = _count_returns_ts(node)
            cyclomatic = branch_count + 1

            # Extract call targets and identifiers
            call_targets = _get_call_targets_ts(node, source_bytes)
            reads, writes = _get_identifiers_ts(node, source_bytes, read_contexts, write_contexts)

            func_info = FunctionInfo(
                name=func_name,
                line_number=start_line,
                line_count=line_count,
                arg_count=param_count,
                avg_arg_length=avg_param_length,
                docstring_length=docstring_length,
                branch_count=branch_count,
                call_count=call_count,
                return_count=return_count,
                cyclomatic_complexity=cyclomatic,
                call_targets=call_targets,
                reads=reads,
                writes=writes,
            )
            functions.append(func_info)

        for child in node.children:
            traverse(child)

    traverse(root)
    return functions


def convert_to_python_ast_style(source_code: str, language: Optional[CodeLanguage] = None) -> ast.Module:
    """
    Parse multi-language code and convert to a Python AST-compatible structure.

    This creates a synthetic Python AST Module containing FunctionDef nodes
    that represent functions from any supported language.
    """
    root, detected_lang = parse_source(source_code, language)
    functions = extract_functions_ts(root, source_code, detected_lang)

    # Create synthetic Python AST
    module = ast.Module(body=[], type_ignores=[])

    for func_info in functions:
        # Create function arguments
        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=f"arg{i}", annotation=None) for i in range(func_info.arg_count)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )

        # Create a synthetic function body with a pass statement
        pass_stmt = ast.Pass(lineno=func_info.line_number)
        pass_stmt.end_lineno = func_info.line_number  # type: ignore
        pass_stmt.col_offset = 0  # type: ignore
        pass_stmt.end_col_offset = 0  # type: ignore
        body = [pass_stmt]

        # Create FunctionDef node
        func_def = ast.FunctionDef(
            name=func_info.name,
            args=args,
            body=body,
            decorator_list=[],
            returns=None,
            lineno=func_info.line_number,
        )

        # Set end_lineno for proper line counting
        end_lineno = func_info.line_number + func_info.line_count - 1
        func_def.end_lineno = end_lineno  # type: ignore
        func_def.col_offset = 0  # type: ignore
        func_def.end_col_offset = 0  # type: ignore

        # Attach our custom metrics as attributes (not standard AST, but we'll use them)
        func_def._custom_metrics = {  # type: ignore
            'line_count': func_info.line_count,
            'arg_count': func_info.arg_count,
            'avg_arg_length': func_info.avg_arg_length,
            'docstring_length': func_info.docstring_length,
            'branch_count': func_info.branch_count,
            'call_count': func_info.call_count,
            'return_count': func_info.return_count,
            'cyclomatic_complexity': func_info.cyclomatic_complexity,
            'call_targets': func_info.call_targets,
            'reads': func_info.reads,
            'writes': func_info.writes,
        }

        module.body.append(func_def)

    return module
