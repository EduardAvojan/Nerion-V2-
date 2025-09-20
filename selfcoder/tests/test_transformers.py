import ast
from textwrap import dedent
from selfcoder.actions import apply_actions_via_ast

def _has_module_logger_boilerplate(tree: ast.Module) -> bool:
    has_import = any(isinstance(n, ast.Import) and any(a.name == "logging" for a in n.names) for n in tree.body)
    has_logger = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "logger" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == "logging"
        and n.value.func.attr == "getLogger"
        for n in tree.body
    )
    return has_import and has_logger

def _get_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"function {name} not found")

def test_try_except_wrapper_injects_boilerplate_and_wraps_body():
    src = dedent('''
    def foo(x, y):
        """add two numbers"""
        return x + y
    ''')
    out = apply_actions_via_ast(src, [
        {"kind": "try_except_wrapper", "action": "try_except_wrapper", "payload": {"function": "foo"}}
    ])
    tree = ast.parse(out)
    assert _has_module_logger_boilerplate(tree)

    fn = _get_func(tree, "foo")
    # first non-docstring statement should be a Try
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(getattr(body[0].value, "value", None), str)):
        body = body[1:]
    assert body, "function body empty after docstring"
    assert isinstance(body[0], ast.Try), "expected try/except wrapper"

    # the except should catch Exception and call logger.exception
    t: ast.Try = body[0]
    assert t.handlers, "no except handlers"
    h = t.handlers[0]
    assert isinstance(h.type, ast.Name) and h.type.id == "Exception"
    # look for logger.exception call inside except body
    has_logger_exception = any(
        isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == "logger"
        and n.value.func.attr in {"exception", "error"}
        for n in h.body
    )
    assert has_logger_exception, "expected logger.exception/error in except"

def test_inject_function_exit_log_adds_exit_line_and_boilerplate():
    src = dedent('''
    def bar(n):
        if n < 0:
            return 0
        return n * 2
    ''')
    out = apply_actions_via_ast(src, [
        {"kind": "inject_function_exit_log", "action": "inject_function_exit_log", "payload": {"function": "bar"}}
    ])
    tree = ast.parse(out)
    assert _has_module_logger_boilerplate(tree)

    fn = _get_func(tree, "bar")
    # find the last statement; allow for appended exit log
    last_stmt = fn.body[-1]
    assert isinstance(last_stmt, ast.Expr) and isinstance(last_stmt.value, ast.Call), "last stmt should be a call"
    call = last_stmt.value
    assert isinstance(call.func, ast.Attribute)
    assert isinstance(call.func.value, ast.Name) and call.func.value.id == "logger"
    assert call.func.attr == "info"
    # message should mention exiting function bar
    arg0 = call.args[0] if call.args else None
    assert isinstance(arg0, ast.Constant) and isinstance(arg0.value, str) and "Exiting" in arg0.value and "bar" in arg0.value