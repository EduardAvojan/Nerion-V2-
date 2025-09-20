"""
Parent Plan Executor
====================

This module executes a ParentDecision plan produced by the Parent LLM.
It is **generic** and has **no references** to engine internals or specific tools.

Usage pattern (from the engine loop):

    executor = ParentExecutor(
        tool_runners={
            "read_url": lambda **kw: run_site_query_from_engine(**kw),
            "web_search": lambda **kw: run_web_search_from_engine(**kw),
            "rename_symbol": lambda **kw: run_rename_symbol_from_engine(**kw),
        },
        ensure_network=lambda required: ensure_network_if_needed(required),
    )

    outcome = executor.execute(decision, user_query)
    # outcome = {"final_text", "action_taken", "success", "error"}

The executor respects the Parent's plan structure without knowing any tool logic.
All side-effects are performed by the injected callables.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Iterable
import time
from pydantic import BaseModel, HttpUrl, ValidationError

from .schemas import ParentDecision


ToolRunner = Callable[..., Any]
EnsureNetwork = Callable[[bool], None]  # should raise or handle permission prompt when required
MetricsHook = Callable[[str, bool, float, Optional[str]], None]
ProgressHook = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]


class _ReadUrlArgs(BaseModel):
    url: HttpUrl
    timeout: Optional[int] = 10


class _WebSearchArgs(BaseModel):
    query: Optional[str]
    max_results: Optional[int] = 5


class _RenameSymbolArgs(BaseModel):
    old: str
    new: str
    simulate: Optional[bool] = True

# Simple placeholder for tools with no args
class _NoArgs(BaseModel):
    pass

class _ReadFileArgs(BaseModel):
    path: str


class ParentExecutor:
    """Execute ParentDecision plans via injected tool runners.

    Parameters
    ----------
    tool_runners : dict[str, ToolRunner]
        Mapping from tool name to a callable that accepts keyword args from `step.args`.
        The callable should return either a string (summary) or a dict payload.
    ensure_network : Callable[[bool], None] | None
        An optional function the executor calls **once** before running steps if the
        decision requires network. If network is not permitted, this function should
        raise an exception (e.g., PermissionError) or otherwise abort.
    """

    def __init__(
        self,
        tool_runners: Dict[str, ToolRunner],
        ensure_network: Optional[EnsureNetwork] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        metrics_hook: Optional[MetricsHook] = None,
        progress_hook: Optional[ProgressHook] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> None:
        self._tools = dict(tool_runners or {})
        self._ensure_net = ensure_network
        self._allowed = set(allowed_tools or self._tools.keys())
        self._metrics = metrics_hook
        self._progress = progress_hook
        self._cancel = cancel_check
        # Built-in validators for common tools
        self._validators: Dict[str, Any] = {
            "read_url": _ReadUrlArgs,
            "web_search": _WebSearchArgs,
            "rename_symbol": _RenameSymbolArgs,
            "run_healthcheck": _NoArgs,
            "run_diagnostics": _NoArgs,
            "list_plugins": _NoArgs,
            "run_pytest_smoke": _NoArgs,
            "read_file": _ReadFileArgs,
            "summarize_file": _ReadFileArgs,
        }

    def execute(self, decision: ParentDecision, user_query: str) -> Dict[str, Any]:
        """Run a ParentDecision plan.

        Returns a dict with keys:
          - final_text: Optional[str]
          - action_taken: Dict[str, Any] (summaries of steps executed)
          - success: bool
          - error: Optional[str]
        """
        action_taken: Dict[str, Any] = {"steps": []}

        # Handle network requirement up-front (single prompt/permission path)
        try:
            if decision.requires_network and self._ensure_net is not None:
                self._ensure_net(True)
        except Exception as e:
            return {
                "final_text": None,
                "action_taken": action_taken,
                "success": False,
                "error": f"network_not_allowed: {e}",
            }

        # Execute each step in order
        try:
            total = len(decision.plan or [])
            had_error_hard = False
            _had_error_soft = False
            first_error_msg: Optional[str] = None
            _any_success = False
            for idx, step in enumerate(decision.plan or []):
                if self._cancel and self._cancel():
                    return {
                        "final_text": None,
                        "action_taken": action_taken,
                        "success": False,
                        "error": "cancelled by user",
                    }
                rec: Dict[str, Any] = {"index": idx, "action": step.action}
                # Progress ack
                try:
                    if step.action == "tool_call":
                        desc = step.summary or (step.tool or "tool")
                    else:
                        desc = step.summary or step.action
                    # Add rationale and cost hint if provided
                    try:
                        if getattr(step, 'why', None):
                            desc = f"{desc} — {step.why}"
                    except Exception:
                        pass
                    try:
                        if getattr(step, 'cost_hint_ms', None) is not None:
                            desc = f"{desc} (~{int(step.cost_hint_ms)}ms)"
                    except Exception:
                        pass
                    print(f"[PLAN] Step {idx+1}/{total}: {desc}")
                    if self._progress:
                        try:
                            self._progress(idx+1, total, desc)
                        except Exception:
                            pass
                except Exception:
                    pass

                if step.action == "ask_user":
                    # Engine should perform the actual ask; we record and stop.
                    rec.update({"summary": step.summary or "ask_user"})
                    action_taken["steps"].append(rec)
                    return {
                        "final_text": None,
                        "action_taken": action_taken,
                        "success": True,
                        "error": None,
                    }

                if step.action == "respond":
                    # Parent wants a direct response; no further tool execution.
                    text = decision.final_response or step.summary or ""
                    rec.update({"summary": "respond"})
                    action_taken["steps"].append(rec)
                    return {
                        "final_text": text,
                        "action_taken": action_taken,
                        "success": True,
                        "error": None,
                    }

                if step.action == "tool_call":
                    tool_name = step.tool or ""
                    runner = self._tools.get(tool_name)
                    if runner is None:
                        # Record and continue rather than abort the whole plan
                        err_msg = f"unknown_tool: {tool_name}"
                        rec.update({"tool": tool_name, "error": err_msg})
                        action_taken["steps"].append(rec)
                        if not bool(getattr(step, 'continue_on_error', False)):
                            had_error_hard = True
                            if first_error_msg is None:
                                first_error_msg = err_msg
                        else:
                            _had_error_soft = True
                        continue
                    if tool_name not in self._allowed:
                        err_msg = f"unauthorized_tool: {tool_name}"
                        rec.update({"tool": tool_name, "error": err_msg})
                        action_taken["steps"].append(rec)
                        if not bool(getattr(step, 'continue_on_error', False)):
                            had_error_hard = True
                            if first_error_msg is None:
                                first_error_msg = err_msg
                        else:
                            _had_error_soft = True
                        continue

                    # Validate args if a model exists
                    args = dict(step.args or {})
                    model = self._validators.get(tool_name)
                    if model is not None:
                        try:
                            args = model(**args).model_dump()  # pydantic v2
                        except ValidationError as ve:
                            err_msg = f"invalid_args for {tool_name}: {ve}"
                            rec.update({"tool": tool_name, "error": err_msg})
                            action_taken["steps"].append(rec)
                            if not bool(getattr(step, 'continue_on_error', False)):
                                had_error_hard = True
                                if first_error_msg is None:
                                    first_error_msg = err_msg
                            else:
                                _had_error_soft = True
                            continue

                    # Execute tool with optional timeout; capture a lightweight summary
                    t0 = time.monotonic()
                    result = None
                    err = None
                    ok = False
                    # Decide timeout and retries with policy defaults only when the field
                    # was not explicitly provided in the Parent step. Pydantic v2 exposes
                    # `model_fields_set`; v1 exposes `__fields_set__`.
                    timeout = None
                    def _field_set(m, name: str) -> bool:
                        try:
                            s = getattr(m, 'model_fields_set', None)
                            if s is not None:
                                return name in s
                        except Exception:
                            pass
                        try:
                            s = getattr(m, '__fields_set__', None)
                            if s is not None:
                                return name in s
                        except Exception:
                            pass
                        return False
                    # Only use step.timeout_s if it was provided; otherwise fill from policy
                    if _field_set(step, 'timeout_s'):
                        try:
                            timeout = float(getattr(step, 'timeout_s')) if getattr(step, 'timeout_s') is not None else None
                        except Exception:
                            timeout = None
                    # Policy defaults for timeout/retry
                    import os as _os
                    policy = (_os.getenv('NERION_POLICY') or '').strip().lower()
                    default_timeout = None
                    default_retry = 0
                    if policy == 'safe':
                        default_timeout = 20.0
                        default_retry = 0
                    elif policy == 'fast':
                        default_timeout = 8.0
                        default_retry = 0
                    else:  # balanced (no default timeout to avoid thread overhead in tests)
                        default_timeout = None
                        default_retry = 1
                    if timeout is None and default_timeout is not None:
                        timeout = default_timeout
                    # Retry: respect explicit step.retry only if it was provided; otherwise use policy default
                    if _field_set(step, 'retry'):
                        retry_count = int(getattr(step, 'retry') or 0)
                    else:
                        retry_count = int(default_retry)
                        # If the step is marked continue_on_error and no explicit retry
                        # was provided, do a single attempt and move on.
                        try:
                            if bool(getattr(step, 'continue_on_error')):
                                retry_count = 0
                        except Exception:
                            pass
                    attempts = max(1, retry_count + 1)
                    backoff = 0.25
                    if timeout and timeout > 0:
                        import concurrent.futures as _f
                        with _f.ThreadPoolExecutor(max_workers=1) as ex:
                            for i in range(attempts):
                                fut = ex.submit(runner, **args)
                                try:
                                    result = fut.result(timeout=timeout)
                                    ok = True
                                    _any_success = True
                                    break
                                except Exception as e:
                                    err = str(e)
                                    ok = False
                                    try:
                                        fut.cancel()
                                    except Exception:
                                        pass
                                    if i < attempts - 1:
                                        time.sleep(backoff)
                                        backoff *= 2
                                        continue
                                    # Exhausted attempts; record error and continue
                                    rec.update({"tool": tool_name, "error": err})
                                    action_taken["steps"].append(rec)
                                    if not bool(getattr(step, 'continue_on_error', False)):
                                        had_error_hard = True
                                        if first_error_msg is None:
                                            first_error_msg = err
                                    else:
                                        _had_error_soft = True
                                    ok = False
                                    break
                    else:
                        for i in range(attempts):
                            try:
                                result = runner(**args)
                                ok = True
                                _any_success = True
                                break
                            except Exception as e:
                                err = str(e)
                                ok = False
                                if i < attempts - 1:
                                    time.sleep(backoff)
                                    backoff *= 2
                                    continue
                                # Exhausted attempts; record error and continue
                                rec.update({"tool": tool_name, "error": err})
                                action_taken["steps"].append(rec)
                                if not bool(getattr(step, 'continue_on_error', False)):
                                    had_error_hard = True
                                    if first_error_msg is None:
                                        first_error_msg = err
                                else:
                                    _had_error_soft = True
                                ok = False
                                break
                    if self._metrics:
                        try:
                            self._metrics(tool_name, ok, time.monotonic() - t0, err)
                        except Exception:
                            pass
                    dur_ms = int((time.monotonic() - t0) * 1000)
                    rec.update({
                        "tool": tool_name,
                        "duration_ms": dur_ms,
                    })
                    if getattr(step, 'why', None):
                        rec["why"] = step.why
                    if getattr(step, 'cost_hint_ms', None) is not None:
                        rec["cost_hint_ms"] = int(step.cost_hint_ms)  # type: ignore[arg-type]
                    if isinstance(result, dict):
                        rec["result"] = {k: result.get(k) for k in list(result)[:6]}
                    else:
                        rec["result"] = str(result)[:500]
                    action_taken["steps"].append(rec)

                else:
                    raise ValueError(f"unsupported_action: {step.action}")

            # If we completed all steps and Parent also provided a final_response, return it
            final_text = decision.final_response
            out = {
                "final_text": final_text,
                "action_taken": action_taken,
                "success": (not had_error_hard),
                "error": (first_error_msg if had_error_hard else None),
            }
            try:
                import os as _os
                import json as _json
                if (_os.getenv('NERION_DEBUG_EXEC') or '').strip():
                    print('[EXEC-DEBUG]', _json.dumps(out, ensure_ascii=False))
            except Exception:
                pass
            return out

        except Exception as e:
            return {
                "final_text": None,
                "action_taken": action_taken,
                "success": False,
                "error": str(e),
            }


def make_default_executor(
    *,
    read_url: Optional[ToolRunner] = None,
    web_search: Optional[ToolRunner] = None,
    rename_symbol: Optional[ToolRunner] = None,
    run_healthcheck: Optional[ToolRunner] = None,
    run_diagnostics: Optional[ToolRunner] = None,
    list_plugins: Optional[ToolRunner] = None,
    run_pytest_smoke: Optional[ToolRunner] = None,
    read_file: Optional[ToolRunner] = None,
    summarize_file: Optional[ToolRunner] = None,
    ensure_network: Optional[EnsureNetwork] = None,
    allowed_tools: Optional[Iterable[str]] = None,
    metrics_hook: Optional[MetricsHook] = None,
    progress_hook: Optional[ProgressHook] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> ParentExecutor:
    """Helper to construct an executor with a common tool set.

    Pass engine-bound callables for each tool you support. Omit any you don't
    want exposed to the Parent yet; missing tools will raise a clear error if
    the Parent attempts to use them.
    """
    runners: Dict[str, ToolRunner] = {}
    if read_url is not None:
        runners["read_url"] = read_url
    if web_search is not None:
        runners["web_search"] = web_search
    if rename_symbol is not None:
        runners["rename_symbol"] = rename_symbol
    if run_healthcheck is not None:
        runners["run_healthcheck"] = run_healthcheck
    if run_diagnostics is not None:
        runners["run_diagnostics"] = run_diagnostics
    if list_plugins is not None:
        runners["list_plugins"] = list_plugins
    if run_pytest_smoke is not None:
        runners["run_pytest_smoke"] = run_pytest_smoke
    if read_file is not None:
        runners["read_file"] = read_file
    if summarize_file is not None:
        runners["summarize_file"] = summarize_file
    return ParentExecutor(
        tool_runners=runners,
        ensure_network=ensure_network,
        allowed_tools=allowed_tools,
        metrics_hook=metrics_hook,
        progress_hook=progress_hook,
        cancel_check=cancel_check,
    )
