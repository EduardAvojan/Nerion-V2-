"""
Inspiration Selector for Autonomous Learning.

This module selects lesson topics based on the CEFR (Common European Framework
of Reference for Languages) adapted for programming education, ranging from
A1 (beginner) to C2 (expert) level.
"""
import random
from typing import Optional


class InspirationSelector:
    """
    Selects inspiration sources for lesson generation based on CEFR levels
    and optional category filtering.

    The CEFR framework provides structured progression:
    - A1/A2: Beginner/Elementary (basic syntax, simple operations)
    - B1/B2: Intermediate/Upper Intermediate (patterns, concurrency, security)
    - C1/C2: Advanced/Expert (distributed systems, advanced algorithms)
    """

    # CEFR-categorized lesson types for structured curriculum
    INSPIRATIONS = [
        # A1 - Beginner
        "a1_variable_scope_errors",
        "a1_type_conversion_bugs",
        "a1_string_concatenation_issues",
        "a1_list_index_errors",
        "a1_basic_syntax_mistakes",
        "a1_simple_logic_errors",
        "a1_function_argument_errors",
        "a1_basic_loop_bugs",
        "a1_dictionary_key_errors",
        "a1_simple_file_operations",
        "a1_basic_exception_handling",
        "a1_simple_input_validation",
        "a1_basic_string_formatting",
        "a1_simple_math_operations",
        "a1_basic_conditional_logic",
        "a1_list_comprehension_basics",

        # A2 - Elementary
        "a2_list_mutation_bugs",
        "a2_dictionary_operations",
        "a2_basic_oop_errors",
        "a2_method_design_issues",
        "a2_simple_inheritance_bugs",
        "a2_file_handling_errors",
        "a2_json_parsing_issues",
        "a2_basic_api_calls",
        "a2_simple_data_validation",
        "a2_basic_testing_patterns",
        "a2_exception_propagation",
        "a2_simple_generators",
        "a2_basic_decorators",
        "a2_context_manager_basics",
        "a2_simple_regex_patterns",
        "a2_basic_async_await",

        # B1 - Intermediate
        "b1_design_pattern_issues",
        "b1_error_handling_strategies",
        "b1_resource_cleanup",
        "b1_algorithm_efficiency",
        "b1_data_structure_selection",
        "b1_testing_strategies",
        "b1_mocking_patterns",
        "b1_dependency_injection",
        "b1_configuration_management",
        "b1_logging_best_practices",
        "b1_database_transactions",
        "b1_query_optimization",
        "b1_api_design_patterns",
        "b1_serialization_issues",
        "b1_caching_basics",
        "b1_retry_mechanisms",

        # B2 - Upper Intermediate
        "b2_race_conditions",
        "b2_deadlock_prevention",
        "b2_memory_leaks",
        "b2_sql_injection_prevention",
        "b2_xss_vulnerabilities",
        "b2_authentication_bypass",
        "b2_n_plus_one_queries",
        "b2_circuit_breaker_pattern",
        "b2_rate_limiting",
        "b2_connection_pooling",
        "b2_distributed_caching",
        "b2_message_queue_patterns",
        "b2_event_driven_architecture",
        "b2_async_concurrency",
        "b2_performance_profiling",
        "b2_security_best_practices",

        # C1 - Advanced (use existing specialized generators)
        "refactoring",
        "bug_fixing",
        "feature_implementation",
        "performance_optimization",
        "code_comprehension",
        "security_hardening",
        "testing_strategies",
        "api_design",
        "concurrency_patterns",
        "data_structures",
        "algorithmic_optimization",
        "error_recovery",
        "database_design",
        "observability",
        "deployment_cicd",
        "caching_strategies",
        "message_queues",
        "resource_management",
        "distributed_systems",
        "scaling_patterns",
        "data_validation",
        "configuration_management",

        # C2 - Expert
        "c2_distributed_locking",
        "c2_consensus_algorithms",

        # HARD CATEGORIES - Semantic Understanding Gaps
        "deadlock_patterns",
        "tainted_data_flow_security",
        "use_after_free_patterns",
        "django_n_plus_one_queries",
        "flask_session_security",
        "pandas_performance_anti_patterns",
        "resource_leak_exception_paths",
        "async_race_conditions",
        "memory_leak_patterns",
        "sql_injection_dataflow",
        "xss_vulnerability_patterns",
        "csrf_protection_patterns",
        "authentication_bypass_patterns",
        "authorization_confusion_patterns",
        "cryptographic_misuse_patterns",
        "timing_attack_vulnerabilities",
        "buffer_overflow_patterns",
        "integer_overflow_patterns",
        "path_traversal_vulnerabilities",
        "command_injection_patterns",
        "c2_eventual_consistency",
        "c2_saga_pattern",
        "c2_cqrs_implementation",
        "c2_event_sourcing",
        "c2_advanced_concurrency",
        "c2_lock_free_algorithms",
        "c2_memory_optimization",
        "c2_cache_coherence",
        "c2_distributed_tracing",
        "c2_service_mesh_patterns",
        "c2_advanced_security",
        "c2_cryptographic_implementations",
        "c2_high_availability_patterns",
        "c2_disaster_recovery",
    ]

    def __init__(self, category_filter: Optional[str] = None):
        """
        Initialize the InspirationSelector.

        Args:
            category_filter: Optional category prefix to filter inspirations
                           (e.g., "a1", "b2", "refactoring")
        """
        self.category_filter = category_filter

    def select(self) -> Optional[str]:
        """
        Selects a source of inspiration based on category filter or random selection.

        Returns:
            Selected inspiration topic, or None if no valid inspirations found
        """
        # If category filter is set, only use inspirations matching that category
        if self.category_filter:
            # Filter inspirations by prefix (e.g., "a1" matches "a1_variable_scope_errors")
            filtered = [
                insp for insp in self.INSPIRATIONS
                if insp.startswith(self.category_filter.lower())
            ]
            if not filtered:
                print(f"[ERROR] No inspirations found for category filter: {self.category_filter}")
                return None
            inspiration = random.choice(filtered)
            print(f"[Inspiration] Using filtered category: {self.category_filter} -> {inspiration}")
        else:
            inspiration = random.choice(self.INSPIRATIONS)
            print(f"[Inspiration] Randomly selected focus: {inspiration}")

        return inspiration

    @classmethod
    def get_all_inspirations(cls) -> list[str]:
        """
        Get the complete list of all available inspirations.

        Returns:
            List of all inspiration topics
        """
        return cls.INSPIRATIONS.copy()

    @classmethod
    def get_inspirations_by_level(cls, level: str) -> list[str]:
        """
        Get inspirations for a specific CEFR level.

        Args:
            level: CEFR level prefix (e.g., "a1", "b2", "c1")

        Returns:
            List of inspirations matching the level
        """
        level_lower = level.lower()
        return [insp for insp in cls.INSPIRATIONS if insp.startswith(level_lower)]
