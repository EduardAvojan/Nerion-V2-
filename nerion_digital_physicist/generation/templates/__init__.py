"""Exports template classes for environment generator."""
from .alg_arithmetic_pipeline import ArithmeticPipelineTemplate
from .bug_off_by_one import OffByOneBugTemplate
from .refactor_duplicate_code import RefactorDuplicateCodeTemplate

__all__ = [
    "ArithmeticPipelineTemplate",
    "OffByOneBugTemplate",
    "RefactorDuplicateCodeTemplate",
]
