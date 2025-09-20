from __future__ import annotations
from pathlib import Path
import re
import ast
from typing import Any, Dict, List, Optional, Callable

def _ensure_logger_boilerplate(src_text):
    tree = ast.parse(src_text)
    need_import = True
    need_logger = True
    for n in tree.body:
        if isinstance(n, ast.Import) and any(a.name == "logging" for a in n.names):
            need_import = False
        if isinstance(n, ast.ImportFrom) and n.module == "logging":
            need_import = False
        if isinstance(n, ast.Assign) and any(getattr(t,'id','')=='logger' for t in n.targets):
            need_logger = False
    pre = []
    if need_import:
        pre.append("import logging")
    if need_logger:
        pre.append("logger = logging.getLogger(__name__)")
    return ("\n".join(pre)+"\n") if pre else ""

def _append_exit_log_to_func(src_text, func_name):
    pre = _ensure_logger_boilerplate(src_text)
    tree = ast.parse(src_text)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == func_name:
            call = ast.parse(f'logger.info("{func_name}: exit")').body[0]
            n.body.append(call)
            break
    body = []
    for n in tree.body:
        body.append(ast.get_source_segment(src_text, n) or "")
    stitched = "".join(body) if body and any(body) else src_text
    if pre:
        stitched = pre + stitched
    return stitched



# --- External transformer registry (for plugins) ---------------------------
# Plugins can register custom transformers that accept (ast.Module, action_dict)
# and return an updated ast.Module.
_EXTERNAL_TRANSFORMERS: Dict[str, Callable[[ast.Module, Dict[str, Any]], ast.Module]] = {}

def register_external_transformer(name: str, fn: Callable[[ast.Module, Dict[str, Any]], ast.Module]) -> None:
    """Register an external AST transformer under a new action kind.

    The callable must accept (tree: ast.Module, action: Dict[str, Any]) and
    return an updated ast.Module. Re-registering the same name overwrites it.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("transformer name must be a non-empty string")
    if not callable(fn):
        raise TypeError("transformer fn must be callable")
    _EXTERNAL_TRANSFORMERS[name] = fn

# --- AST helpers -----------------------------------------------------------

class ModuleDocstringAdder(ast.NodeTransformer):
    """Ensure the module has a top-level docstring. If already present, no-op."""

    def __init__(self, doc: str) -> None:
        self.doc = doc

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(getattr(body[0].value, "value", None), str)
        ):
            return node  # already has a module docstring
        body.insert(0, ast.Expr(value=ast.Constant(value=self.doc)))
        node.body = body
        return node


class FunctionDocstringAdder(ast.NodeTransformer):
    """Add a docstring to a given function if it has none."""

    def __init__(self, function: str, doc: str) -> None:
        super().__init__()
        self.function = function
        self.doc = doc

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.function:
            return self.generic_visit(node)
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(getattr(body[0].value, "value", None), str)
        ):
            return self.generic_visit(node)  # already has a docstring
        body.insert(0, ast.Expr(value=ast.Constant(value=self.doc)))
        node.body = body
        return self.generic_visit(node)

def _ensure_logging_boilerplate(node: ast.Module) -> None:
    """Ensure that 'import logging' and 'logger = logging.getLogger(__name__)' are present."""
    has_import_logging = False
    has_logger_assignment = False

    for stmt in node.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "logging":
                    has_import_logging = True
                    break
        elif isinstance(stmt, ast.Assign):
            if (
                len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "logger"
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and isinstance(stmt.value.func.value, ast.Name)
                and stmt.value.func.value.id == "logging"
                and stmt.value.func.attr == "getLogger"
            ):
                has_logger_assignment = True
        if has_import_logging and has_logger_assignment:
            break

    # Determine insertion point: after a module docstring if present,
    # and after any leading imports we add.
    insert_pos = 0
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(getattr(node.body[0], "value", None), ast.Constant)
        and isinstance(getattr(node.body[0].value, "value", None), str)
    ):
        insert_pos = 1  # keep module docstring first

    if not has_import_logging:
        import_logging = ast.Import(names=[ast.alias(name="logging", asname=None)])
        node.body.insert(insert_pos, import_logging)
        insert_pos += 1

    if not has_logger_assignment:
        getLogger_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="logging", ctx=ast.Load()),
                attr="getLogger",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id="__name__", ctx=ast.Load())],
            keywords=[],
        )
        logger_assign = ast.Assign(
            targets=[ast.Name(id="logger", ctx=ast.Store())],
            value=getLogger_call,
        )
        node.body.insert(insert_pos, logger_assign)

class FunctionEntryLogger(ast.NodeTransformer):
    """Insert a logger.info call at the start of a specified function."""

    def __init__(self, function: str) -> None:
        super().__init__()
        self.function = function

    def visit_Module(self, node: ast.Module):
        _ensure_logging_boilerplate(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.function:
            return self.generic_visit(node)

        # Determine insertion point after docstring if present
        start_idx = 0
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(getattr(node.body[0].value, "value", None), str)
        ):
            start_idx = 1

        # Avoid duplicate logger at the insertion point
        if (
            len(node.body) > start_idx
            and isinstance(node.body[start_idx], ast.Expr)
            and isinstance(node.body[start_idx].value, ast.Call)
            and isinstance(node.body[start_idx].value.func, ast.Attribute)
            and isinstance(node.body[start_idx].value.func.value, ast.Name)
            and node.body[start_idx].value.func.value.id == "logger"
        ):
            return self.generic_visit(node)

        log_msg = f"Entering function {self.function}"
        log_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="logger", ctx=ast.Load()),
                    attr="info",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=log_msg)],
                keywords=[],
            )
        )
        node.body.insert(start_idx, log_call)
        return self.generic_visit(node)

class TryExceptWrapperTransformer(ast.NodeTransformer):
    """Wrap the body of a specified function in a try/except Exception block, logging and re-raising."""

    def __init__(self, function: str) -> None:
        super().__init__()
        self.function = function

    def visit_Module(self, node: ast.Module):
        _ensure_logging_boilerplate(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.function:
            return self.generic_visit(node)

        # Check if already wrapped in try/except Exception at top level
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Try)
        ):
            try_node = node.body[0]
            handler_found = False
            for handler in try_node.handlers:
                if (
                    handler.type is not None
                    and isinstance(handler.type, ast.Name)
                    and handler.type.id == "Exception"
                ):
                    handler_found = True
                    break
            if handler_found:
                # Already wrapped, skip
                return self.generic_visit(node)

        # Wrap the entire function body in try/except Exception
        original_body = node.body
        except_handler = ast.ExceptHandler(
            type=ast.Name(id="Exception", ctx=ast.Load()),
            name="e",
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="logger", ctx=ast.Load()),
                            attr="exception",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Constant(value=f"Exception in function {self.function}:"),
                        ],
                        keywords=[],
                    )
                ),
                ast.Raise(exc=None, cause=None),
            ],
        )
        try_node = ast.Try(
            body=original_body,
            handlers=[except_handler],
            orelse=[],
            finalbody=[],
        )
        node.body = [try_node]
        return self.generic_visit(node)

class FunctionExitLogger(ast.NodeTransformer):
    def __init__(self, function: str) -> None:
        super().__init__()
        self.function = function

    def visit_Module(self, node: ast.Module):
        _ensure_logging_boilerplate(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.function:
            return self.generic_visit(node)

        log_msg = f"Exiting function {self.function}"
        end_msg = f"Exiting function {self.function} (end)"

        def _is_log_with_msg(stmt: ast.stmt, msg: str) -> bool:
            return (
                isinstance(stmt, ast.Expr)
                and isinstance(getattr(stmt, "value", None), ast.Call)
                and isinstance(getattr(stmt.value, "func", None), ast.Attribute)
                and isinstance(getattr(stmt.value.func, "value", None), ast.Name)
                and stmt.value.func.value.id == "logger"
                and stmt.value.func.attr == "info"
                and len(stmt.value.args) == 1
                and isinstance(stmt.value.args[0], ast.Constant)
                and stmt.value.args[0].value == msg
            )

        # Detect existing logs to make the transform idempotent
        has_exit = any(_is_log_with_msg(s, log_msg) for s in node.body)
        has_end = any(_is_log_with_msg(s, end_msg) for s in node.body)

        # Insert the exit log immediately BEFORE the final return, if any and if missing
        if not has_exit:
            last_ret_idx = None
            for i in range(len(node.body) - 1, -1, -1):
                if isinstance(node.body[i], ast.Return):
                    last_ret_idx = i
                    break
            if last_ret_idx is not None:
                exit_log_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="logger", ctx=ast.Load()),
                            attr="info",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=log_msg)],
                        keywords=[],
                    )
                )
                node.body.insert(last_ret_idx, exit_log_call)

        # Ensure the last statement is a logging call (with a distinct message), if missing
        if not has_end:
            end_log_call = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="logger", ctx=ast.Load()),
                        attr="info",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=end_msg)],
                    keywords=[],
                )
            )
            node.body.append(end_log_call)

        return self.generic_visit(node)

# --- FunctionInserter and ClassInserter -------------------------------------
class FunctionInserter(ast.NodeTransformer):
    """Insert a new function definition if missing.

    Payload keys supported:
      - name / function: function name (required)
      - args: list[str] of parameter names (optional)
      - doc / docstring: optional docstring for the function
    """
    def __init__(self, name: str, args: Optional[List[str]] = None, doc: Optional[str] = None) -> None:
        super().__init__()
        self.name = name
        self.args = args or []
        self.doc = doc

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        # If function already exists, no-op
        for n in node.body:
            if isinstance(n, ast.FunctionDef) and n.name == self.name:
                return node

        # Build arguments node (posonly/kwonly kept empty for simplicity)
        args_node = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=a) for a in self.args],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        body: List[ast.stmt] = []
        if self.doc:
            body.append(ast.Expr(value=ast.Constant(value=self.doc)))
        body.append(ast.Pass())
        fn = ast.FunctionDef(
            name=self.name,
            args=args_node,
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        node.body.append(fn)
        return node

class ClassInserter(ast.NodeTransformer):
    """Insert a new class definition if missing.

    Payload keys supported:
      - name / class: class name (required)
      - bases: list[str] of base class names (optional)
      - doc / docstring: optional class docstring
    """
    def __init__(self, name: str, bases: Optional[List[str]] = None, doc: Optional[str] = None) -> None:
        super().__init__()
        self.name = name
        self.bases = bases or []
        self.doc = doc

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        # If class already exists, no-op
        for n in node.body:
            if isinstance(n, ast.ClassDef) and n.name == self.name:
                return node

        bases_nodes = [ast.Name(id=b, ctx=ast.Load()) for b in self.bases]
        body: List[ast.stmt] = []
        if self.doc:
            body.append(ast.Expr(value=ast.Constant(value=self.doc)))
        body.append(ast.Pass())
        cls = ast.ClassDef(
            name=self.name,
            bases=bases_nodes,
            keywords=[],
            body=body,
            decorator_list=[],
        )
        node.body.append(cls)
        return node
class ImportNormalizer(ast.NodeTransformer):  # noqa: D401
    """Normalize imports: remove duplicates and sort groups (import/from)."""
    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        imports: list[ast.stmt] = []
        others: list[ast.stmt] = []
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)
            else:
                others.append(stmt)
        if not imports:
            return node
        seen = set()
        imps: list[ast.stmt] = []
        for stmt in imports:
            if isinstance(stmt, ast.Import):
                uniq = []
                for alias in stmt.names:
                    key = (alias.name, alias.asname or '')
                    if key not in seen:
                        seen.add(key)
                        uniq.append(alias)
                if uniq:
                    imps.append(ast.Import(names=uniq))
            else:
                imps.append(stmt)
        imp_only = [s for s in imps if isinstance(s, ast.Import)]
        from_only = [s for s in imps if isinstance(s, ast.ImportFrom)]
        imp_only.sort(key=lambda s: s.names[0].name if isinstance(s, ast.Import) and s.names else '')
        from_only.sort(key=lambda s: (s.module or '') if isinstance(s, ast.ImportFrom) else '')
        node.body = imp_only + from_only + others
        return node

class LoggingInjector(ast.NodeTransformer):  # noqa: D401
    """Placeholder for a logging injector transformer (no-op)."""
    pass

class TryExceptWrapper(ast.NodeTransformer):  # noqa: D401
    """Placeholder for a try/except wrapper transformer (no-op)."""
    pass

class RetryAdder(ast.NodeTransformer):  # noqa: D401
    """Placeholder for a retry-adder transformer (no-op)."""
    pass

class PromoteConstant(ast.NodeTransformer):  # noqa: D401
    """Placeholder for a constant promotion transformer (no-op)."""
    pass


class RenameFunctionSafe(ast.NodeTransformer):  # noqa: D401
    """Placeholder for a safe function rename transformer (no-op)."""
    pass


class RenameParamTransformer(ast.NodeTransformer):
    """Rename a parameter inside a function and its in-body references.

    Payload: {"function": str, "old": str, "new": str}
    """
    def __init__(self, function: str, old: str, new: str) -> None:
        super().__init__()
        self.function = function
        self.old = old
        self.new = new

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        if node.name != self.function:
            return self.generic_visit(node)
        for a in node.args.args:
            if a.arg == self.old:
                a.arg = self.new
        for a in getattr(node.args, 'kwonlyargs', []) or []:
            if a.arg == self.old:
                a.arg = self.new
        if node.args.vararg and node.args.vararg.arg == self.old:
            node.args.vararg.arg = self.new
        if node.args.kwarg and node.args.kwarg.arg == self.old:
            node.args.kwarg.arg = self.new
        class _Renamer(ast.NodeTransformer):
            def __init__(self, old: str, new: str):
                self.old = old
                self.new = new
            def visit_Name(self, n: ast.Name):  # type: ignore[override]
                if n.id == self.old:
                    n.id = self.new
                return n
        node.body = _Renamer(self.old, self.new).visit(ast.Module(body=node.body)).body  # type: ignore
        return self.generic_visit(node)


class RenameCallKeyword(ast.NodeTransformer):
    """Rename a keyword argument name for calls to a target function in-module."""
    def __init__(self, function: str, old: str, new: str) -> None:
        super().__init__()
        self.function = function
        self.old = old
        self.new = new

    def visit_Call(self, node: ast.Call):  # type: ignore[override]
        try:
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == self.function:
                for kw in node.keywords or []:
                    if kw.arg == self.old:
                        kw.arg = self.new
        except Exception:
            pass
        return self.generic_visit(node)


# --- SimplifyBranching transformer ------------------------------------------
class SimplifyBranching(ast.NodeTransformer):
    """Simplify nested/positive-first branching into guard clauses for a target function.

    Heuristics:
      - If a function starts with `if <name>:` and has an `else:` block, convert to
        `if not <name>:` guard with the else-body, followed by the then-body.
      - If no explicit function name is provided, pick the function containing a
        given `line` (1-based), else fall back to the first function in the module.
    The transform is conservative and idempotent-ish; if patterns do not match,
    the function is left unchanged.
    """

    def __init__(self, function: Optional[str] = None, line: Optional[int] = None) -> None:
        super().__init__()
        self.function = function
        self.line = line
        self._target: Optional[str] = None

    # Utilities -------------------------------------------------------------
    def _function_contains_line(self, node: ast.FunctionDef, line: int) -> bool:
        try:
            start = getattr(node, 'lineno', None)
            end = getattr(node, 'end_lineno', None)
            if start is None:
                return False
            if end is None:
                # Best-effort: treat until next node; assume contains
                return line >= start
            return start <= line <= end
        except Exception:
            return False

    def _ends_with_return(self, stmts: List[ast.stmt]) -> bool:
        for i in range(len(stmts) - 1, -1, -1):
            s = stmts[i]
            # skip pass or docstring exprs
            if isinstance(s, ast.Pass):
                continue
            if isinstance(s, ast.Expr) and isinstance(getattr(s, 'value', None), ast.Constant) and isinstance(getattr(s.value, 'value', None), str):
                continue
            return isinstance(s, ast.Return)
        return False

    def _flatten_elses_with_return(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        """For any `if ...:` whose body ends in a `return`, hoist the `else:` body after the `if` and drop the `else:`.
        Recurse into nested blocks conservatively.
        """
        out: List[ast.stmt] = []
        for s in stmts:
            if isinstance(s, ast.If):
                # Recurse first
                s.body = self._flatten_elses_with_return(s.body)
                s.orelse = self._flatten_elses_with_return(s.orelse)
                if s.orelse and self._ends_with_return(s.body):
                    # Hoist else-branch statements after the if; remove else
                    hoist = s.orelse
                    s.orelse = []
                    out.append(s)
                    out.extend(hoist)
                else:
                    out.append(s)
            else:
                out.append(s)
        return out

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        # Resolve target function name once if not provided
        if self.function is None and self.line is not None:
            for n in node.body:
                if isinstance(n, ast.FunctionDef) and self._function_contains_line(n, self.line):
                    self._target = n.name
                    break
        if self.function is None and self._target is None:
            # fallback to first function
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    self._target = n.name
                    break
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        target_name = self.function or self._target
        if target_name and node.name != target_name:
            return self.generic_visit(node)

        # Pattern: top-level If with test `Name` (truthy) and an orelse present
        if not node.body or not isinstance(node.body[0], ast.If):
            # After guard-clause promotion, run a pass that hoists else-bodies when the then-branch returns
            node.body = self._flatten_elses_with_return(node.body)
            ast.fix_missing_locations(node)
            return self.generic_visit(node)

        top_if: ast.If = node.body[0]
        # Ensure we have an orelse to promote to a guard
        if not top_if.orelse:
            node.body = self._flatten_elses_with_return(node.body)
            ast.fix_missing_locations(node)
            return self.generic_visit(node)

        # Accept tests of the form `Name` (truthy) or `UnaryOp(Not, Name)` (negated)
        test = top_if.test
        if isinstance(test, ast.Name):
            cond_name = test.id
            make_guard_negated = True  # `if data:` -> `if not data:` guard
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Name):
            cond_name = test.operand.id
            make_guard_negated = False  # `if not data:` already a guard
        else:
            node.body = self._flatten_elses_with_return(node.body)
            ast.fix_missing_locations(node)
            return self.generic_visit(node)

        # Build new guarded body
        new_body: List[ast.stmt] = []
        # Preserve module/function docstring at start of function
        start_idx = 0
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], 'value', None), ast.Constant)
            and isinstance(getattr(node.body[0].value, 'value', None), str)
        ):
            start_idx = 1

        # Only transform when the If is exactly at start_idx
        if node.body and start_idx < len(node.body) and node.body[start_idx] is top_if:
            # Construct guard: if not <cond_name>: <orelse>
            guard_test: ast.expr
            if make_guard_negated:
                guard_test = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=cond_name, ctx=ast.Load()))
            else:
                guard_test = top_if.test  # already `not <name>`

            guard_if = ast.If(test=guard_test, body=top_if.orelse, orelse=[])

            # New function body: (optional docstring already accounted for) + guard + then-body
            prefix = node.body[:start_idx]
            then_body = top_if.body
            suffix = node.body[start_idx+1:]  # statements after the top-if
            new_body = list(prefix) + [guard_if] + then_body + suffix
            node.body = new_body
            node.body = self._flatten_elses_with_return(node.body)
            ast.fix_missing_locations(node)
            return self.generic_visit(node)

        # After guard-clause promotion, run a pass that hoists else-bodies when the then-branch returns
        node.body = self._flatten_elses_with_return(node.body)
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


def _apply_single_action(tree: ast.Module, action: dict) -> ast.Module:
    """Apply a single normalized action dict to the AST module and return the updated tree.

    Recognized kinds:
      - add_module_docstring
      - add_function_docstring
      - inject_function_entry_log
      - inject_function_exit_log
      - try_except_wrapper
      - insert_function
      - insert_class
      - block (sequencing; applies nested `actions` in order)
      - ensure_test (no-op at AST level; handled by orchestrator)
    Unrecognized kinds are no-ops.
    """
    kind = action.get("kind") or action.get("action")
    payload = action.get("payload") or {}

    # Filesystem-only actions are handled outside the AST pipeline
    if kind == "ensure_test":
        return tree

    if kind == "add_module_docstring":
        doc = payload.get("doc") or payload.get("docstring") or ""
        if isinstance(doc, str) and doc:
            return (ModuleDocstringAdder(doc).visit(tree) or tree)  # type: ignore[return-value]
        return tree

    if kind == "add_function_docstring":
        fn = payload.get("function")
        doc = payload.get("doc") or payload.get("docstring") or ""
        if isinstance(fn, str) and fn and isinstance(doc, str) and doc:
            return (FunctionDocstringAdder(fn, doc).visit(tree) or tree)  # type: ignore[return-value]
        return tree

    if kind == "inject_function_entry_log":
        fn = payload.get("function")
        if isinstance(fn, str) and fn:
            return (FunctionEntryLogger(fn).visit(tree) or tree)  # type: ignore[return-value]
        return tree

    if kind == "inject_function_exit_log":
        fn = payload.get("function")
        if isinstance(fn, str) and fn:
            return (FunctionExitLogger(fn).visit(tree) or tree)  # type: ignore[return-value]
        return tree

    if kind == "try_except_wrapper":
        fn = payload.get("function")
        if isinstance(fn, str) and fn:
            return (TryExceptWrapperTransformer(fn).visit(tree) or tree)  # type: ignore[return-value]
        return tree

    if kind == "simplify_branching":
        fn = payload.get("function")
        line = payload.get("line")
        try:
            line_int = int(line) if line is not None else None
        except Exception:
            line_int = None
        return (SimplifyBranching(function=fn, line=line_int).visit(tree) or tree)  # type: ignore[return-value]

    if kind == "insert_function":
        name = payload.get("name") or payload.get("function")
        args = payload.get("args") or []
        doc = payload.get("doc") or payload.get("docstring")
        if isinstance(name, str) and name:
            return (FunctionInserter(name, args=args, doc=doc).visit(tree) or tree)  # type: ignore[return-value]
        return tree

    if kind == "insert_class":
        name = payload.get("name") or payload.get("class")
        bases = payload.get("bases") or []
        doc = payload.get("doc") or payload.get("docstring")
        if isinstance(name, str) and name:
            return (ClassInserter(name, bases=bases, doc=doc).visit(tree) or tree)  # type: ignore[return-value]
        return tree

    if kind == "block":
        # Sequence nested actions against the same tree
        subactions = action.get("actions") or payload.get("actions") or []
        if isinstance(subactions, list):
            t = tree
            for sub in subactions:
                if isinstance(sub, dict):
                    t = _apply_single_action(t, sub)
            return t
        return tree

    if kind == "rename_param":
        fn = payload.get("function")
        old = payload.get("old")
        new = payload.get("new")
        if isinstance(fn, str) and fn and old and new:
            return (RenameParamTransformer(str(fn), str(old), str(new)).visit(tree) or tree)
        return tree

    if kind == "rename_call_kwarg":
        fn = payload.get("function")
        old = payload.get("old")
        new = payload.get("new")
        if isinstance(fn, str) and fn and old and new:
            return (RenameCallKeyword(str(fn), str(old), str(new)).visit(tree) or tree)
        return tree

    if kind == "normalize_imports":
        return (ImportNormalizer().visit(tree) or tree)

    # External/custom kinds via plugin registry
    ext = _EXTERNAL_TRANSFORMERS.get(str(kind or ""))
    if ext is not None:
        try:
            return ext(tree, action) or tree
        except Exception:
            return tree

    # No-op for unknown kinds
    return tree

def _orig_apply_actions_via_ast(source, actions):
    try:
        tree = ast.parse(source)
    except Exception:
        return source
    try:
        for action in actions or []:
            if isinstance(action, dict):
                tree = _apply_single_action(tree, action)
        ast.fix_missing_locations(tree)
        try:
            return ast.unparse(tree)
        except Exception:
            try:
                from ast import unparse as _unparse
                return _unparse(tree)
            except Exception:
                return source
    except Exception:
        return source

def apply_actions_via_ast(src_text, actions):
    updated = src_text
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        k = a.get("kind") or a.get("action")
        payload = a.get("payload") or {}

        # Map legacy/unsupported actions to a supported transformer
        if k == "extract_function":
            k = "simplify_branching"

        # Normalize payload keys
        norm_payload = dict(payload)
        if "docstring" in norm_payload and "doc" not in norm_payload:
            norm_payload["doc"] = norm_payload["docstring"]
        # Accept generic content key for doc-bearing actions
        if "content" in norm_payload and "doc" not in norm_payload and k in ("add_module_docstring", "add_function_docstring", "insert_function", "insert_class"):
            norm_payload["doc"] = norm_payload["content"]

        # Preserve extra top-level keys (for plugins) while normalizing kind/payload
        norm_action = dict(a)
        norm_action["kind"] = k
        norm_action["payload"] = norm_payload

        # If this is a top-level block with nested actions, expand them inline
        nested = None
        if isinstance(a, dict):
            nested = a.get("actions") or norm_action.get("actions")
        if isinstance(nested, list) and (norm_action.get("kind") in ("block", None)):
            for sub in nested:
                if isinstance(sub, dict):
                    updated = _orig_apply_actions_via_ast(updated, [sub])
            continue

        updated = _orig_apply_actions_via_ast(updated, [norm_action])
    # Ensure POSIX-friendly trailing newline
    if not updated.endswith("\n"):
        updated += "\n"
    return updated


# --- Test scaffolding utility -----------------------------------------------
def build_test_scaffold(src_path: str, symbol: str, kind: str = "function") -> tuple[Path, str]:
    """Compute tests/test_<module>.py path and content for a minimal pytest scaffold.
    Returns (path, content). Caller is responsible for writing.
    Idempotent: if an existing file already has our test for `symbol`, returns existing content.
    """
    src = Path(src_path)
    modname = src.stem
    tests_dir = src.parent / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / f"test_{modname}.py"

    header = "import pytest\nimport importlib.util, pathlib\n\n"
    loader = (
        "def _load():\n"
        "    p = pathlib.Path('" + src.as_posix() + "')\n"
        "    spec = importlib.util.spec_from_file_location(p.stem, p)\n"
        "    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "    return m\n\n"
    )

    if kind == "function":
        test_name = f"test_{symbol}_exists"
    else:
        test_name = f"test_{symbol}_symbol_exists"

    body = (
        f"def {test_name}():\n"
        f"    m = _load()\n"
        f"    assert hasattr(m, '{symbol}')\n"
    )

    new_content = header + loader + body + "\n"
    if test_file.exists():
        existing = test_file.read_text(encoding="utf-8")
        if re.search(rf"def {re.escape(test_name)}\(\):", existing):
            return test_file, existing
        return test_file, (existing.rstrip() + "\n\n" + body + "\n")
    else:
        return test_file, new_content
