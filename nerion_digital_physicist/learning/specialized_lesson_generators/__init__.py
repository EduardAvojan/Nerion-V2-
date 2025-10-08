"""
Specialized Lesson Generators Package.

This package contains all specialized generator methods for programming
lesson topics across all CEFR levels (A1-C2), organized by domain for
better maintainability.

Structure:
- code_quality: Refactoring, testing, API design
- security: Security hardening, data validation
- performance: Performance optimization, caching, algorithms
- infrastructure: Deployment, observability, messaging
- systems: Distributed systems, scaling, concurrency, databases
- general: Bug fixing, features, code comprehension
- cefr: CEFR-categorized lesson generation (handles A1-C2 levels)
- cefr_prompts: CEFR prompt data (imported by cefr module)
"""

from .code_quality import CodeQualityGenerators
from .security import SecurityGenerators
from .performance import PerformanceGenerators
from .infrastructure import InfrastructureGenerators
from .systems import SystemsGenerators
from .general import GeneralGenerators
from .cefr import CEFRGenerator


class SpecializedLessonGenerators:
    """
    Unified interface to all specialized lesson generators.

    This class aggregates all domain-specific generator classes and delegates
    method calls to the appropriate generator. Handles both specialized topics
    and CEFR-categorized lessons across all levels (A1-C2).
    """

    def __init__(self):
        """Initialize all specialized generators."""
        self.code_quality = CodeQualityGenerators()
        self.security = SecurityGenerators()
        self.performance = PerformanceGenerators()
        self.infrastructure = InfrastructureGenerators()
        self.systems = SystemsGenerators()
        self.general = GeneralGenerators()
        self.cefr = CEFRGenerator()

    # Delegate to code_quality generators
    def _generate_refactoring_idea(self, *args, **kwargs):
        return self.code_quality._generate_refactoring_idea(*args, **kwargs)

    def _generate_testing_idea(self, *args, **kwargs):
        return self.code_quality._generate_testing_idea(*args, **kwargs)

    def _generate_api_design_idea(self, *args, **kwargs):
        return self.code_quality._generate_api_design_idea(*args, **kwargs)

    # Delegate to security generators
    def _generate_security_idea(self, *args, **kwargs):
        return self.security._generate_security_idea(*args, **kwargs)

    def _generate_data_validation_idea(self, *args, **kwargs):
        return self.security._generate_data_validation_idea(*args, **kwargs)

    # Delegate to performance generators
    def _generate_performance_optimization_idea(self, *args, **kwargs):
        return self.performance._generate_performance_optimization_idea(*args, **kwargs)

    def _generate_caching_idea(self, *args, **kwargs):
        return self.performance._generate_caching_idea(*args, **kwargs)

    def _generate_algorithm_idea(self, *args, **kwargs):
        return self.performance._generate_algorithm_idea(*args, **kwargs)

    # Delegate to infrastructure generators
    def _generate_deployment_idea(self, *args, **kwargs):
        return self.infrastructure._generate_deployment_idea(*args, **kwargs)

    def _generate_observability_idea(self, *args, **kwargs):
        return self.infrastructure._generate_observability_idea(*args, **kwargs)

    def _generate_message_queue_idea(self, *args, **kwargs):
        return self.infrastructure._generate_message_queue_idea(*args, **kwargs)

    # Delegate to systems generators
    def _generate_concurrency_idea(self, *args, **kwargs):
        return self.systems._generate_concurrency_idea(*args, **kwargs)

    def _generate_data_structure_idea(self, *args, **kwargs):
        return self.systems._generate_data_structure_idea(*args, **kwargs)

    def _generate_error_recovery_idea(self, *args, **kwargs):
        return self.systems._generate_error_recovery_idea(*args, **kwargs)

    def _generate_database_idea(self, *args, **kwargs):
        return self.systems._generate_database_idea(*args, **kwargs)

    def _generate_resource_management_idea(self, *args, **kwargs):
        return self.systems._generate_resource_management_idea(*args, **kwargs)

    def _generate_distributed_systems_idea(self, *args, **kwargs):
        return self.systems._generate_distributed_systems_idea(*args, **kwargs)

    def _generate_scaling_idea(self, *args, **kwargs):
        return self.systems._generate_scaling_idea(*args, **kwargs)

    # Delegate to general generators
    def _generate_bug_fix_idea(self, *args, **kwargs):
        return self.general._generate_bug_fix_idea(*args, **kwargs)

    def _generate_feature_implementation_idea(self, *args, **kwargs):
        return self.general._generate_feature_implementation_idea(*args, **kwargs)

    def _generate_code_comprehension_idea(self, *args, **kwargs):
        return self.general._generate_code_comprehension_idea(*args, **kwargs)

    def _generate_configuration_idea(self, *args, **kwargs):
        return self.general._generate_configuration_idea(*args, **kwargs)

    # Delegate to CEFR generator
    def _generate_cefr_idea(self, *args, **kwargs):
        return self.cefr._generate_cefr_idea(*args, **kwargs)


__all__ = ["SpecializedLessonGenerators"]
