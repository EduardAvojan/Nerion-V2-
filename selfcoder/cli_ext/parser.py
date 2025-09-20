"""CLI parser extender for selfcoder (modular subcommand registration).

This wires subcommands split out from `selfcoder.cli` (healthcheck, plan, deps, docs, self-improve).
Additional subcommands can be moved here over time.
"""
from __future__ import annotations


def extend_parser(subparsers) -> None:
    """Allow external modules to register their subcommands."""
    # Keep CLI resilient if optional components are missing
    try:
        from .commands import health as _health
        _health.register(subparsers)
    except Exception:
        pass
    try:
        from .commands import plan as _plan
        _plan.register(subparsers)
    except Exception:
        pass
    try:
        from .commands import deps_cli as _deps
        _deps.register(subparsers)
    except Exception:
        pass
    try:
        from . import deps_cli as _deps2
        _deps2.register(subparsers)
    except Exception:
        pass
    try:
        from .commands import docs_cli as _docs
        _docs.register(subparsers)
    except Exception:
        pass
    try:
        from . import docs_cli as _docs2
        _docs2.register(subparsers)
    except Exception:
        pass
    try:
        from .commands import self_improve as _self_improve_cmd
        _self_improve_cmd.register(subparsers)
    except Exception:
        pass
    try:
        from . import self_improve as _self_improve
        _self_improve.register(subparsers)
    except Exception:
        pass