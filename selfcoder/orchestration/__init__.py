"""
Orchestration utilities package.

This package provides modular utilities for code orchestration including:
- Environment configuration
- Action validation
- Symbol and import analysis
- Test utilities and impact prediction
- AST transformation
- Diff/preview generation
- Code healers
- Filesystem actions

All utilities are organized by domain for better maintainability.
"""

# Environment utilities
from .environment import (
    env_true,
    prepare_for_prompt,
)

# Validation utilities
from .validators import (
    should_skip,
    validate_actions,
    split_fs_and_ast_actions,
    file_exists,
)

# Symbol analysis utilities
from .symbol_analyzer import (
    symbol_present_in_file,
    evaluate_preconditions,
    extract_import_module_names,
    module_resolves,
    unresolved_imports_in_file,
    REPO_ROOT,
)

# Test utilities
from .test_utils import (
    tests_collect_ok,
    causal_from_post_failures,
    predict_impacted_tests,
)

# AST transformation utilities
from .ast_transformer import (
    apply_ast_actions_transactional,
    apply_actions_preview,
    run_ast_actions,
    dry_run_orchestrate,
)

# Diff/preview utilities
from .diff_preview import (
    unified_diff_for_file,
    preview_bundle,
)

# Healer utilities
from .healers import (
    ALLOWED_HEALERS,
    healer_format,
    healer_isort,
    run_healers,
)

# Filesystem action utilities
from .fs_actions import (
    apply_fs_actions,
)

# Main runner functions
from .runner import (
    run_actions_on_file,
    run_actions_on_files,
    apply_plan,
    OrchestrateResult,
    _normalize_actions_for_apply,
    run_batch_actions_on_files,
)

__all__ = [
    # Environment
    "env_true",
    "prepare_for_prompt",
    # Validation
    "should_skip",
    "validate_actions",
    "split_fs_and_ast_actions",
    "file_exists",
    # Symbol analysis
    "symbol_present_in_file",
    "evaluate_preconditions",
    "extract_import_module_names",
    "module_resolves",
    "unresolved_imports_in_file",
    "REPO_ROOT",
    # Test utilities
    "tests_collect_ok",
    "causal_from_post_failures",
    "predict_impacted_tests",
    # AST transformation
    "apply_ast_actions_transactional",
    "apply_actions_preview",
    "run_ast_actions",
    "dry_run_orchestrate",
    # Diff/preview
    "unified_diff_for_file",
    "preview_bundle",
    # Healers
    "ALLOWED_HEALERS",
    "healer_format",
    "healer_isort",
    "run_healers",
    # Filesystem actions
    "apply_fs_actions",
    # Main runner functions
    "run_actions_on_file",
    "run_actions_on_files",
    "apply_plan",
    "OrchestrateResult",
    "_normalize_actions_for_apply",
    "run_batch_actions_on_files",
]
