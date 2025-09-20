"""Actions package public API: thin re-exports.

This module intentionally **does not** import from `app.*` to avoid cycles.
It provides a stable surface by re-exporting the actual implementations from
`selfcoder.actions.transformers` and optional helpers from sibling modules.
"""
from __future__ import annotations

# Re-export core AST utilities and compatibility placeholders from transformers
from .transformers import (
    apply_actions_via_ast,
    ModuleDocstringAdder,
    FunctionDocstringAdder,
    ImportNormalizer,
    LoggingInjector,
    TryExceptWrapper,
    RetryAdder,
    PromoteConstant,
    RenameFunctionSafe,
)

# Optional cross-file helpers: import if available, otherwise provide no-ops
try:  # Prefer real implementations if present
    from .crossfile import (
        ImportFromRename,  # type: ignore[F401]
        ModuleAttributeRename,  # type: ignore[F401]
        apply_crossfile_rename,  # type: ignore[F401]
    )
except Exception:  # pragma: no cover - graceful fallback if module missing
    class ImportFromRename:  # noqa: D401
        """Fallback placeholder for cross-file import renamer."""
        pass

    class ModuleAttributeRename:  # noqa: D401
        """Fallback placeholder for module attribute renamer."""
        pass

    def apply_crossfile_rename(*_args, **_kwargs):  # noqa: D401
        """Fallback no-op for cross-file rename; returns False to indicate no change."""
        return False

# Optional: tests generator re-export
try:
    from .tests_gen import generate_unit_test_file  # type: ignore[F401]
except Exception:  # pragma: no cover
    def generate_unit_test_file(*_args, **_kwargs):  # noqa: D401
        """Fallback unit-test generator: returns an empty string."""
        return ""

# Public export list so re-exports are recognized by linters (ruff F401).
__all__ = [
    # Core patcher and AST helpers
    "apply_actions_via_ast",
    "ModuleDocstringAdder",
    "FunctionDocstringAdder",
    # Compatibility placeholder transformers (no-ops)
    "ImportNormalizer",
    "LoggingInjector",
    "TryExceptWrapper",
    "RetryAdder",
    "PromoteConstant",
    "RenameFunctionSafe",
    # Cross-file ops
    "ImportFromRename",
    "ModuleAttributeRename",
    "apply_crossfile_rename",
    # Tests generation
    "generate_unit_test_file",
]
