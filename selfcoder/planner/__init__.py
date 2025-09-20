

"""Planner subpackage entrypoints.

This package hosts the planning layer for Nerion Selfcoder.  It intentionally
keeps imports lightweight so that simply importing :mod:`selfcoder.planner`
never explodes even while the concrete planner evolves.

Exports here are *best-effort* re-exports from :mod:`selfcoder.planner.planner`.
If that module is missing (e.g., during incremental refactors), imports will
still succeed and the public API will appear as an empty surface.
"""
from __future__ import annotations

# Best-effort re-exports (kept lazy/guarded to avoid import-time failures)
try:  # pragma: no cover - defensive wrapper
    from .planner import plan_from_text, PLAN_VERSION  # type: ignore
    __all__ = [
        "plan_from_text",
        "PLAN_VERSION",
    ]
except Exception:  # pragma: no cover
    # If the concrete planner isn't available yet, expose an empty surface so
    # that `import selfcoder.planner` remains safe and tests can still run.
    __all__: list[str] = []