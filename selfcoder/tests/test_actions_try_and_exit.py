

# selfcoder/tests/test_actions_try_and_exit.py
from __future__ import annotations
import ast
from textwrap import dedent

from selfcoder.actions import apply_actions_via_ast


def _count_try_nodes_in_func(src: str, func_name: str) -> int:
    """Parse src and count ast.Try nodes inside a given function."""
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == func_name:
            return sum(1 for s in ast.walk(n) if isinstance(s, ast.Try))
    return 0


def test_inject_function_exit_log_before_return_and_idempotent():
    src = dedent(
        '''
        def run_smoke() -> bool:
            """doc"""
            return True
        '''
    ).strip()

    # 1) Apply once — should import logging + create logger + insert log BEFORE the final return
    out1 = apply_actions_via_ast(
        src,
        [{"kind": "inject_function_exit_log", "payload": {"function": "run_smoke"}}],
    )
    assert out1 != src

    # Must add import + logger boilerplate
    assert "import logging" in out1
    assert "logger = logging.getLogger(__name__)" in out1

    # The exit log must appear before the final return
    r_idx = out1.index("return True")
    if "logger.info('Exiting function run_smoke')" in out1:
        log_idx = out1.index("logger.info('Exiting function run_smoke')")
    else:
        log_idx = out1.index('logger.info("Exiting function run_smoke")')
    assert log_idx < r_idx, "exit log should be inserted before final return"

    # 2) Apply again — no duplication (idempotent)
    out2 = apply_actions_via_ast(
        out1,
        [{"kind": "inject_function_exit_log", "payload": {"function": "run_smoke"}}],
    )
    # The log line should appear exactly once
    count_single = (
        out2.count("logger.info('Exiting function run_smoke')")
        + out2.count('logger.info("Exiting function run_smoke")')
    )
    assert count_single == 1, "exit log should not duplicate on re-apply"


def test_try_except_wrapper_wraps_and_reraises_and_is_idempotent():
    src = dedent(
        '''
        def will_boom(x):
            y = x + 1
            raise ValueError("boom")
        '''
    ).strip()

    # 1) Apply wrapper
    out1 = apply_actions_via_ast(
        src,
        [{"kind": "try_except_wrapper", "payload": {"function": "will_boom"}}],
    )
    assert out1 != src

    # Must add import + logger boilerplate
    assert "import logging" in out1
    assert "logger = logging.getLogger(__name__)" in out1

    # Confirm AST has exactly one Try in the function
    assert _count_try_nodes_in_func(out1, "will_boom") == 1

    # 2) Executing the function still raises (re-raises) the original exception
    ns: dict = {}
    exec(out1, ns, ns)
    try:
        ns["will_boom"](0)
        raised = False
    except Exception as e:
        raised = True
        # We expect a ValueError with "boom" because we re-raise the original
        assert isinstance(e, Exception)
        assert "boom" in str(e)
    assert raised, "wrapped function should re-raise the original exception"

    # 3) Idempotency: re-apply wrapper again should not add another nested try
    out2 = apply_actions_via_ast(
        out1,
        [{"kind": "try_except_wrapper", "payload": {"function": "will_boom"}}],
    )
    assert _count_try_nodes_in_func(out2, "will_boom") == 1