# /Users/ed/Nerion/selfcoder/__init__.py
"""
Lightweight selfcoder package exports.

Keep this file minimal to avoid circular imports. Only re-export what external
code needs.
"""

__version__ = "0.0.1"

from .types import EditAction, EditPlan  # convenience re-exports

# Only the transformer hook is required by app/nerion_autocoder.py
from .actions.transformers import apply_actions_via_ast  # noqa: F401


# Optional: available for future use; lazy import to avoid circular imports
def apply_crossfile_rename(*args, **kwargs):
    from .actions.crossfile import apply_crossfile_rename as real_apply_crossfile_rename
    return real_apply_crossfile_rename(*args, **kwargs)

__all__ = [
    "EditAction",
    "EditPlan",
    "apply_actions_via_ast",
    "apply_crossfile_rename",
    "__version__",
]