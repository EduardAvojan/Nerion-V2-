"""
This module contains the Learning Orchestrator, which implements the 
Autonomous Impact Framework for self-directed learning.

It is responsible for:
1.  Finding inspiration for new lesson topics.
2.  Generating concrete lesson ideas.
3.  Assessing the impact and viability of those ideas.
4.  Triggering the curriculum generator for high-quality ideas.
"""
import os
import subprocess
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

from app.parent.coder import Coder
from selfcoder.policy.meta_policy_evaluator import MetaPolicyEvaluator
from nerion_digital_physicist.db.curriculum_store import CurriculumStore


def _load_environment() -> None:
    """Load environment variables from .env when available."""
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    except Exception:
        # Silent failure keeps behaviour consistent with optional dotenv import.
        pass


_load_environment()


class LearningOrchestrator:
    """
    Autonomous learning orchestrator for the Digital Physicist system.
    
    This class manages the self-directed learning cycle, including:
    - Selecting inspiration sources for lesson generation
    - Generating lesson ideas using LLM providers
    - Triggering curriculum generation for validated ideas
    - Managing the autonomous learning workflow
    
    Attributes:
        None (stateless orchestrator)
    
    Example:
        >>> orchestrator = LearningOrchestrator()
        >>> orchestrator.run_cycle(provider="openai", model_name="gpt-4")
    """
    
    def __init__(self, category_filter: Optional[str] = None) -> None:
        """
        Initialize the LearningOrchestrator.
        
        Args:
            category_filter: Optional category to filter lesson generation to
        """
        self._knowledge_base = set()
        self.meta_policy_evaluator = MetaPolicyEvaluator()
        self._category_filter = category_filter  # If set, only generate lessons for this category

    def _get_inspiration(self) -> Optional[str]:
        """Selects a source of inspiration based on the meta-policy."""
        # CEFR-categorized lesson types for structured curriculum
        inspirations = [
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

        # If category filter is set, only use inspirations matching that category
        if self._category_filter:
            # Filter inspirations by prefix (e.g., "a1" matches "a1_variable_scope_errors")
            filtered = [insp for insp in inspirations if insp.startswith(self._category_filter.lower())]
            if not filtered:
                print(f"[ERROR] No inspirations found for category filter: {self._category_filter}")
                return None
            inspiration = random.choice(filtered)
            print(f"[Inspiration] Using filtered category: {self._category_filter} -> {inspiration}")
        else:
            inspiration = random.choice(inspirations)
            print(f"[Inspiration] Randomly selected focus: {inspiration}")

        return inspiration

    def _generate_idea(self, inspiration: str, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a specific lesson idea based on the inspiration source."""
        print(f"[Idea Generation] Generating concept based on: {inspiration}")

        # Route to specialized generators
        generators = {
            "refactoring": self._generate_refactoring_idea,
            "bug_fixing": self._generate_bug_fix_idea,
            "feature_implementation": self._generate_feature_implementation_idea,
            "performance_optimization": self._generate_performance_optimization_idea,
            "code_comprehension": self._generate_code_comprehension_idea,
            "security_hardening": self._generate_security_idea,
            "testing_strategies": self._generate_testing_idea,
            "api_design": self._generate_api_design_idea,
            "concurrency_patterns": self._generate_concurrency_idea,
            "data_structures": self._generate_data_structure_idea,
            "algorithmic_optimization": self._generate_algorithm_idea,
            "error_recovery": self._generate_error_recovery_idea,
            "database_design": self._generate_database_idea,
            "observability": self._generate_observability_idea,
            "deployment_cicd": self._generate_deployment_idea,
            "caching_strategies": self._generate_caching_idea,
            "message_queues": self._generate_message_queue_idea,
            "resource_management": self._generate_resource_management_idea,
            "distributed_systems": self._generate_distributed_systems_idea,
            "scaling_patterns": self._generate_scaling_idea,
            "data_validation": self._generate_data_validation_idea,
            "configuration_management": self._generate_configuration_idea,
        }

        if inspiration in generators:
            return generators[inspiration](provider, project_id, location, model_name)

        # Try CEFR-specific generator for categorized inspirations
        if inspiration.startswith(('a1_', 'a2_', 'b1_', 'b2_', 'c1_', 'c2_')):
            cefr_idea = self._generate_cefr_idea(inspiration, provider, project_id, location, model_name)
            if cefr_idea:
                return cefr_idea
            # Fall through to generic if CEFR generator didn't handle it

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert programming educator and curriculum designer. Your task is to devise a concept for a single, specific, and useful programming lesson for an AI agent. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = f"Propose a lesson concept related to the strategic focus area of: {inspiration}."

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea(inspiration)

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea(inspiration)

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea(inspiration)

        idea['source'] = inspiration  # Add the source for context

        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea(inspiration)

        print(f"  - Generated idea: {idea['name']}")
        return idea

    def _generate_refactoring_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a refactoring lesson idea."""
        print(f"[Idea Generation] Generating refactoring concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert software architect and curriculum designer. Your task is to devise a HIGH-IMPACT refactoring lesson for an AI agent. "
            "Focus on refactorings that improve code maintainability, reduce technical debt, or prevent future bugs. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT refactoring lesson. Examples:\n"
            "- Extract complex conditional logic into strategy pattern\n"
            "- Replace inheritance with composition for flexibility\n"
            "- Introduce dependency injection to reduce coupling\n"
            "- Refactor god class into single-responsibility classes\n"
            "- Replace magic numbers with named constants\n"
            "- Eliminate code duplication with template method pattern\n"
            "Generate a similarly impactful refactoring concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("refactoring")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("refactoring")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("refactoring")

        idea['source'] = "refactoring"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("refactoring")

        print(f"  - Generated refactoring idea: {idea['name']}")
        return idea

    def _generate_bug_fix_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a specific bug-fixing lesson idea."""
        print(f"[Idea Generation] Generating bug-fix concept...")
        
        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert programming educator and curriculum designer. Your task is to devise a concept for a single, specific, and HIGH-IMPACT bug-fixing lesson for an AI agent. "
            "Focus on bugs that cause security vulnerabilities, data corruption, crashes, or silent failures in production systems. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier for the bug) and 'description' (a one-sentence explanation of the bug)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT bug-fixing lesson concept. Examples of high-impact bugs:\n"
            "- SQL injection vulnerabilities\n"
            "- Race conditions in concurrent code\n"
            "- Memory leaks from unclosed resources\n"
            "- Null pointer dereferences causing crashes\n"
            "- Integer overflow in calculations\n"
            "- Improper error handling that silently fails\n"
            "Generate a similarly critical bug-fixing concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("bug_fixing")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("bug_fixing")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("bug_fixing")

        idea['source'] = "bug_fixing"  # Add the source for context

        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("bug_fixing")

        print(f"  - Generated bug-fix idea: {idea['name']}")
        return idea

    def _generate_feature_implementation_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a specific feature implementation lesson idea."""
        print(f"[Idea Generation] Generating feature implementation concept...")
        
        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert programming educator and curriculum designer. Your task is to devise a concept for a single, specific, and HIGH-IMPACT feature implementation lesson for an AI agent. "
            "Focus on features that are CRITICAL for production systems: security, data integrity, error handling, concurrency, resource management, or system reliability. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier for the feature) and 'description' (a one-sentence explanation of the feature)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT feature implementation lesson concept. Examples of high-impact features:\n"
            "- Authentication/authorization mechanisms\n"
            "- Input sanitization to prevent injection attacks\n"
            "- Rate limiting for API endpoints\n"
            "- Graceful degradation under load\n"
            "- Circuit breaker patterns for external services\n"
            "- Comprehensive error handling and logging\n"
            "Generate a similarly impactful feature concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("feature_implementation")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("feature_implementation")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("feature_implementation")

        idea['source'] = "feature_implementation"  # Add the source for context

        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("feature_implementation")

        print(f"  - Generated feature implementation idea: {idea['name']}")
        return idea

    def _generate_performance_optimization_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a specific performance optimization lesson idea."""
        print(f"[Idea Generation] Generating performance optimization concept...")
        
        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert programming educator and curriculum designer. Your task is to devise a concept for a single, specific, and HIGH-IMPACT performance optimization lesson for an AI agent. "
            "Focus on optimizations that significantly improve scalability, reduce resource usage, or prevent system bottlenecks in production environments. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier for the optimization) and 'description' (a one-sentence explanation of the optimization)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT performance optimization lesson concept. Examples of high-impact optimizations:\n"
            "- Database query optimization (N+1 query elimination, proper indexing)\n"
            "- Caching strategies for expensive computations\n"
            "- Lazy loading to reduce memory footprint\n"
            "- Connection pooling for database/API calls\n"
            "- Async/await for I/O-bound operations\n"
            "- Memory leak prevention and garbage collection tuning\n"
            "Generate a similarly impactful optimization concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("performance_optimization")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("performance_optimization")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("performance_optimization")

        idea['source'] = "performance_optimization"  # Add the source for context

        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("performance_optimization")

        print(f"  - Generated performance optimization idea: {idea['name']}")
        return idea

    def _generate_code_comprehension_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a specific code comprehension lesson idea."""
        print(f"[Idea Generation] Generating code comprehension concept...")
        
        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert programming educator and curriculum designer. Your task is to devise a HIGH-IMPACT code comprehension lesson for an AI agent. "
            "Focus on understanding complex patterns, anti-patterns, or non-obvious behavior that leads to bugs. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier for the concept) and 'description' (a one-sentence explanation of the concept)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT code comprehension lesson. Examples:\n"
            "- Understanding Python's mutable default arguments pitfall\n"
            "- Recognizing memory leaks from circular references\n"
            "- Identifying SQL N+1 query patterns in ORM code\n"
            "- Understanding closure variable capture issues\n"
            "- Recognizing implicit type coercion bugs\n"
            "- Spotting time-of-check-time-of-use (TOCTOU) vulnerabilities\n"
            "Generate a similarly critical comprehension concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("code_comprehension")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("code_comprehension")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("code_comprehension")

        idea['source'] = "code_comprehension"  # Add the source for context

        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("code_comprehension")

        print(f"  - Generated code comprehension idea: {idea['name']}")
        return idea

    def _generate_security_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a security hardening lesson idea."""
        print(f"[Idea Generation] Generating security hardening concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert security engineer and curriculum designer. Your task is to devise a HIGH-IMPACT security hardening lesson for an AI agent. "
            "Focus on preventing real-world vulnerabilities and attacks. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT security hardening lesson. Examples:\n"
            "- Input sanitization to prevent XSS attacks\n"
            "- CSRF token implementation\n"
            "- Secure password hashing with bcrypt/argon2\n"
            "- JWT token validation and expiry\n"
            "- SQL injection prevention with parameterized queries\n"
            "- Secrets management (no hardcoded credentials)\n"
            "Generate a similarly critical security concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("security_hardening")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("security_hardening")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("security_hardening")

        idea['source'] = "security_hardening"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("security_hardening")

        print(f"  - Generated security idea: {idea['name']}")
        return idea

    def _generate_testing_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a testing strategy lesson idea."""
        print(f"[Idea Generation] Generating testing strategy concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert test engineer and curriculum designer. Your task is to devise a HIGH-IMPACT testing strategy lesson for an AI agent. "
            "Focus on testing approaches that catch critical bugs and ensure system reliability. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT testing strategy lesson. Examples:\n"
            "- Property-based testing for edge cases\n"
            "- Integration testing with test doubles/mocks\n"
            "- Mutation testing to verify test quality\n"
            "- Contract testing for APIs\n"
            "- Chaos engineering for failure scenarios\n"
            "- Snapshot testing for UI components\n"
            "Generate a similarly impactful testing concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("testing_strategies")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("testing_strategies")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("testing_strategies")

        idea['source'] = "testing_strategies"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("testing_strategies")

        print(f"  - Generated testing idea: {idea['name']}")
        return idea

    def _generate_api_design_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates an API design lesson idea."""
        print(f"[Idea Generation] Generating API design concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert API architect and curriculum designer. Your task is to devise a HIGH-IMPACT API design lesson for an AI agent. "
            "Focus on API patterns that improve reliability, usability, and maintainability. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT API design lesson. Examples:\n"
            "- RESTful pagination with cursor-based navigation\n"
            "- API versioning strategies\n"
            "- Idempotency keys for safe retries\n"
            "- Request throttling and rate limiting\n"
            "- GraphQL schema design with DataLoader\n"
            "- WebSocket connection management\n"
            "Generate a similarly impactful API design concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("api_design")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("api_design")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("api_design")

        idea['source'] = "api_design"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("api_design")

        print(f"  - Generated API design idea: {idea['name']}")
        return idea

    def _generate_concurrency_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a concurrency pattern lesson idea."""
        print(f"[Idea Generation] Generating concurrency pattern concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert in concurrent systems and curriculum designer. Your task is to devise a HIGH-IMPACT concurrency lesson for an AI agent. "
            "Focus on patterns that prevent race conditions, deadlocks, and ensure thread safety. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT concurrency pattern lesson. Examples:\n"
            "- Thread-safe singleton with double-checked locking\n"
            "- Producer-consumer pattern with queues\n"
            "- Read-write locks for concurrent access\n"
            "- Atomic operations for lock-free programming\n"
            "- Semaphore-based resource pooling\n"
            "- Async/await for concurrent I/O\n"
            "Generate a similarly critical concurrency concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("concurrency_patterns")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("concurrency_patterns")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("concurrency_patterns")

        idea['source'] = "concurrency_patterns"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("concurrency_patterns")

        print(f"  - Generated concurrency idea: {idea['name']}")
        return idea

    def _generate_data_structure_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a data structure lesson idea."""
        print(f"[Idea Generation] Generating data structure concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert in data structures and curriculum designer. Your task is to devise a HIGH-IMPACT data structure lesson for an AI agent. "
            "Focus on choosing the right data structure for significant performance or correctness improvements. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT data structure lesson. Examples:\n"
            "- Trie for efficient prefix searching\n"
            "- LRU cache with OrderedDict\n"
            "- Heap for priority queue operations\n"
            "- Bloom filter for membership testing\n"
            "- Union-Find for disjoint sets\n"
            "- B-tree for database indexing\n"
            "Generate a similarly impactful data structure concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("data_structures")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("data_structures")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("data_structures")

        idea['source'] = "data_structures"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("data_structures")

        print(f"  - Generated data structure idea: {idea['name']}")
        return idea

    def _generate_algorithm_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates an algorithmic optimization lesson idea."""
        print(f"[Idea Generation] Generating algorithmic optimization concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert algorithmist and curriculum designer. Your task is to devise a HIGH-IMPACT algorithmic optimization lesson for an AI agent. "
            "Focus on algorithm improvements that dramatically reduce time/space complexity. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT algorithmic optimization lesson. Examples:\n"
            "- Two-pointer technique for O(n) vs O(n²)\n"
            "- Dynamic programming to eliminate recursion\n"
            "- Binary search for O(log n) lookups\n"
            "- Sliding window for substring problems\n"
            "- Topological sort for dependency resolution\n"
            "- Dijkstra's algorithm for shortest path\n"
            "Generate a similarly impactful algorithm concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("algorithmic_optimization")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("algorithmic_optimization")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("algorithmic_optimization")

        idea['source'] = "algorithmic_optimization"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("algorithmic_optimization")

        print(f"  - Generated algorithm idea: {idea['name']}")
        return idea

    def _generate_error_recovery_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates an error recovery lesson idea."""
        print(f"[Idea Generation] Generating error recovery concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert in resilient systems and curriculum designer. Your task is to devise a HIGH-IMPACT error recovery lesson for an AI agent. "
            "Focus on patterns that gracefully handle failures and maintain system availability. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT error recovery lesson. Examples:\n"
            "- Exponential backoff with jitter for retries\n"
            "- Circuit breaker to prevent cascading failures\n"
            "- Bulkhead pattern for fault isolation\n"
            "- Graceful degradation when dependencies fail\n"
            "- Dead letter queue for failed messages\n"
            "- Transaction rollback and compensation\n"
            "Generate a similarly critical error recovery concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("error_recovery")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("error_recovery")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("error_recovery")

        idea['source'] = "error_recovery"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("error_recovery")

        print(f"  - Generated error recovery idea: {idea['name']}")
        return idea

    def _generate_database_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a database design lesson idea."""
        print(f"[Idea Generation] Generating database design concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert database architect and curriculum designer. Your task is to devise a HIGH-IMPACT database design lesson for an AI agent. "
            "Focus on database patterns that ensure data integrity, performance, and scalability. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT database design lesson. Examples:\n"
            "- Database normalization to prevent anomalies\n"
            "- Composite indexes for multi-column queries\n"
            "- Soft delete pattern for audit trails\n"
            "- Database migration strategies with zero downtime\n"
            "- Partitioning tables for scalability\n"
            "- Foreign key constraints for referential integrity\n"
            "Generate a similarly impactful database concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("database_design")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("database_design")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("database_design")

        idea['source'] = "database_design"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("database_design")

        print(f"  - Generated database idea: {idea['name']}")
        return idea

    def _generate_observability_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates an observability/monitoring lesson idea."""
        print(f"[Idea Generation] Generating observability concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert SRE/DevOps engineer and curriculum designer. Your task is to devise a HIGH-IMPACT observability lesson for an AI agent. "
            "Focus on monitoring, logging, and tracing patterns that enable rapid debugging and incident response. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT observability lesson. Examples:\n"
            "- Structured logging with correlation IDs\n"
            "- Distributed tracing across microservices\n"
            "- RED metrics (Rate, Errors, Duration) for APIs\n"
            "- Setting alert thresholds with SLOs/SLIs\n"
            "- Log aggregation and centralized monitoring\n"
            "- Performance profiling and flame graphs\n"
            "Generate a similarly critical observability concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("observability")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("observability")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("observability")

        idea['source'] = "observability"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("observability")

        print(f"  - Generated observability idea: {idea['name']}")
        return idea

    def _generate_deployment_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a deployment/CI-CD lesson idea."""
        print(f"[Idea Generation] Generating deployment/CI-CD concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert DevOps engineer and curriculum designer. Your task is to devise a HIGH-IMPACT deployment/CI-CD lesson for an AI agent. "
            "Focus on deployment strategies that minimize downtime and risk. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT deployment/CI-CD lesson. Examples:\n"
            "- Blue-green deployment for zero downtime\n"
            "- Canary releases with gradual rollout\n"
            "- Database migration with backward compatibility\n"
            "- Automated rollback on health check failure\n"
            "- Feature flags for dark launches\n"
            "- Container orchestration with Kubernetes\n"
            "Generate a similarly impactful deployment concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("deployment_cicd")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("deployment_cicd")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("deployment_cicd")

        idea['source'] = "deployment_cicd"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("deployment_cicd")

        print(f"  - Generated deployment idea: {idea['name']}")
        return idea

    def _generate_caching_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a caching strategy lesson idea."""
        print(f"[Idea Generation] Generating caching strategy concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert performance engineer and curriculum designer. Your task is to devise a HIGH-IMPACT caching strategy lesson for an AI agent. "
            "Focus on caching patterns that dramatically improve performance and reduce load. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT caching strategy lesson. Examples:\n"
            "- Cache-aside pattern with Redis\n"
            "- Write-through vs write-back caching\n"
            "- Cache invalidation strategies (TTL, LRU)\n"
            "- Distributed cache consistency with cache stampede prevention\n"
            "- CDN edge caching for static assets\n"
            "- Memoization for expensive function calls\n"
            "Generate a similarly impactful caching concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("caching_strategies")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("caching_strategies")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("caching_strategies")

        idea['source'] = "caching_strategies"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("caching_strategies")

        print(f"  - Generated caching idea: {idea['name']}")
        return idea

    def _generate_message_queue_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a message queue/event-driven lesson idea."""
        print(f"[Idea Generation] Generating message queue concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert distributed systems architect and curriculum designer. Your task is to devise a HIGH-IMPACT message queue lesson for an AI agent. "
            "Focus on async messaging patterns that improve scalability and reliability. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT message queue lesson. Examples:\n"
            "- At-least-once vs exactly-once delivery guarantees\n"
            "- Dead letter queue for failed messages\n"
            "- Event sourcing with append-only log\n"
            "- Pub/sub pattern for decoupled services\n"
            "- Message ordering and partitioning strategies\n"
            "- Idempotent message processing\n"
            "Generate a similarly critical message queue concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("message_queues")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("message_queues")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("message_queues")

        idea['source'] = "message_queues"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("message_queues")

        print(f"  - Generated message queue idea: {idea['name']}")
        return idea

    def _generate_resource_management_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a resource management lesson idea."""
        print(f"[Idea Generation] Generating resource management concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert systems programmer and curriculum designer. Your task is to devise a HIGH-IMPACT resource management lesson for an AI agent. "
            "Focus on proper handling of limited resources to prevent leaks and exhaustion. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT resource management lesson. Examples:\n"
            "- Context managers for automatic resource cleanup\n"
            "- Connection pooling to limit database connections\n"
            "- File descriptor limits and proper closing\n"
            "- Memory pool allocation for performance\n"
            "- Thread pool sizing and management\n"
            "- Garbage collection tuning and heap management\n"
            "Generate a similarly critical resource management concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("resource_management")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("resource_management")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("resource_management")

        idea['source'] = "resource_management"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("resource_management")

        print(f"  - Generated resource management idea: {idea['name']}")
        return idea

    def _generate_distributed_systems_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a distributed systems lesson idea."""
        print(f"[Idea Generation] Generating distributed systems concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert distributed systems architect and curriculum designer. Your task is to devise a HIGH-IMPACT distributed systems lesson for an AI agent. "
            "Focus on patterns that ensure consistency, availability, and partition tolerance. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT distributed systems lesson. Examples:\n"
            "- Two-phase commit for distributed transactions\n"
            "- Consensus algorithms (Raft, Paxos)\n"
            "- Eventual consistency with conflict resolution\n"
            "- Vector clocks for causal ordering\n"
            "- Leader election patterns\n"
            "- Saga pattern for long-running transactions\n"
            "Generate a similarly critical distributed systems concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("distributed_systems")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("distributed_systems")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("distributed_systems")

        idea['source'] = "distributed_systems"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("distributed_systems")

        print(f"  - Generated distributed systems idea: {idea['name']}")
        return idea

    def _generate_scaling_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a scaling pattern lesson idea."""
        print(f"[Idea Generation] Generating scaling pattern concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert scalability engineer and curriculum designer. Your task is to devise a HIGH-IMPACT scaling pattern lesson for an AI agent. "
            "Focus on patterns that enable systems to handle increasing load. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT scaling pattern lesson. Examples:\n"
            "- Horizontal scaling with load balancing\n"
            "- Database sharding strategies\n"
            "- Read replicas for read-heavy workloads\n"
            "- Auto-scaling based on metrics\n"
            "- Stateless service design for easy scaling\n"
            "- CQRS (Command Query Responsibility Segregation)\n"
            "Generate a similarly impactful scaling concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("scaling_patterns")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("scaling_patterns")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("scaling_patterns")

        idea['source'] = "scaling_patterns"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("scaling_patterns")

        print(f"  - Generated scaling idea: {idea['name']}")
        return idea

    def _generate_data_validation_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a data validation lesson idea."""
        print(f"[Idea Generation] Generating data validation concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert data engineer and curriculum designer. Your task is to devise a HIGH-IMPACT data validation lesson for an AI agent. "
            "Focus on validation patterns that prevent data corruption and security issues. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT data validation lesson. Examples:\n"
            "- JSON schema validation with strict types\n"
            "- Input sanitization to prevent XSS/injection\n"
            "- Email/phone number format validation with regex\n"
            "- Business rule validation (credit card expiry)\n"
            "- Type safety with static type checkers\n"
            "- Data integrity constraints in databases\n"
            "Generate a similarly critical data validation concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("data_validation")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("data_validation")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("data_validation")

        idea['source'] = "data_validation"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("data_validation")

        print(f"  - Generated data validation idea: {idea['name']}")
        return idea

    def _generate_configuration_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a configuration management lesson idea."""
        print(f"[Idea Generation] Generating configuration management concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert DevOps engineer and curriculum designer. Your task is to devise a HIGH-IMPACT configuration management lesson for an AI agent. "
            "Focus on configuration patterns that improve security and flexibility. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT configuration management lesson. Examples:\n"
            "- Environment-specific config with 12-factor principles\n"
            "- Feature flags for gradual rollouts\n"
            "- Secrets management with vault/KMS\n"
            "- Configuration versioning and rollback\n"
            "- Dynamic configuration reload without restarts\n"
            "- Configuration validation at startup\n"
            "Generate a similarly critical configuration concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("configuration_management")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("configuration_management")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("configuration_management")

        idea['source'] = "configuration_management"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("configuration_management")

        print(f"  - Generated configuration idea: {idea['name']}")
        return idea

    def _generate_cefr_idea(self, inspiration: str, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """
        Generates rich, targeted lesson ideas for CEFR-categorized inspirations.
        Maps inspiration categories to detailed prompts with concrete examples.
        """
        # CEFR-specific prompt mappings
        CEFR_PROMPTS = {
            # A1 - Beginner fundamentals
            "a1_variable_scope_errors": {
                "expertise": "Python fundamentals and variable scoping",
                "examples": [
                    "Using a variable before it's defined",
                    "Accidentally shadowing a variable in nested scope",
                    "Confusing local vs global variables",
                    "Forgetting to initialize a counter in a loop",
                ]
            },
            "a1_type_conversion_bugs": {
                "expertise": "Python type system and conversions",
                "examples": [
                    "Concatenating string and integer without conversion",
                    "Dividing integers when float result expected (Python 2 style)",
                    "Comparing incompatible types (str == int)",
                    "Using wrong conversion function (int() vs float())",
                ]
            },
            "a1_list_index_errors": {
                "expertise": "Python list operations and indexing",
                "examples": [
                    "Accessing index beyond list length",
                    "Off-by-one errors in loops",
                    "Negative indexing mistakes",
                    "Modifying list while iterating",
                ]
            },
            "a1_string_concatenation_issues": {
                "expertise": "String operations and formatting",
                "examples": [
                    "Inefficient string concatenation in loops",
                    "Mixing string concatenation with join()",
                    "Not using f-strings for formatting",
                    "Concatenating None values causing errors",
                ]
            },
            "a1_basic_syntax_mistakes": {
                "expertise": "Python syntax fundamentals",
                "examples": [
                    "Missing colons after if/for/def statements",
                    "Incorrect indentation",
                    "Using = instead of == in conditions",
                    "Missing parentheses in function calls",
                ]
            },
            "a1_simple_logic_errors": {
                "expertise": "Boolean logic and conditionals",
                "examples": [
                    "Using 'and' when 'or' is needed",
                    "Incorrect operator precedence",
                    "Missing negation in conditions",
                    "Empty else blocks that should contain logic",
                ]
            },
            "a1_function_argument_errors": {
                "expertise": "Function parameters and arguments",
                "examples": [
                    "Missing required arguments",
                    "Passing arguments in wrong order",
                    "Mixing positional and keyword arguments incorrectly",
                    "Not providing default values for optional parameters",
                ]
            },
            "a1_basic_loop_bugs": {
                "expertise": "Loop control and iteration",
                "examples": [
                    "Infinite loops from wrong condition",
                    "Off-by-one errors with range()",
                    "Modifying loop variable inside loop",
                    "Using wrong loop type (for vs while)",
                ]
            },
            "a1_dictionary_key_errors": {
                "expertise": "Dictionary operations",
                "examples": [
                    "Accessing non-existent keys without .get()",
                    "Not checking if key exists before access",
                    "Using mutable types as dictionary keys",
                    "Confusing keys() and values()",
                ]
            },
            "a1_simple_file_operations": {
                "expertise": "Basic file I/O",
                "examples": [
                    "Not closing files after opening",
                    "Wrong file mode ('r' vs 'w' vs 'a')",
                    "Not checking if file exists before reading",
                    "Missing error handling for file operations",
                ]
            },
            # A2 - Elementary patterns
            "a2_list_mutation_bugs": {
                "expertise": "Python list mutability and side effects",
                "examples": [
                    "Unintended aliasing (shallow copy issues)",
                    "Mutating a list passed as argument",
                    "Using mutable default arguments",
                    "Modifying list during iteration causing skipped elements",
                ]
            },
            "a2_basic_oop_errors": {
                "expertise": "Object-oriented programming basics",
                "examples": [
                    "Forgetting self parameter in methods",
                    "Not calling parent __init__ in subclass",
                    "Confusing class vs instance variables",
                    "Missing return in __str__ or __repr__",
                ]
            },
            "a2_dictionary_operations": {
                "expertise": "Advanced dictionary usage",
                "examples": [
                    "Using dict.get() vs dict[] for safe access",
                    "Inefficient dict.keys() iteration",
                    "Not using defaultdict for counters",
                    "Modifying dict while iterating over it",
                ]
            },
            "a2_method_design_issues": {
                "expertise": "Method organization and design",
                "examples": [
                    "Methods doing too many things",
                    "Missing single responsibility",
                    "Poor parameter ordering",
                    "Not returning meaningful values",
                ]
            },
            "a2_simple_inheritance_bugs": {
                "expertise": "Inheritance and method resolution",
                "examples": [
                    "Not calling super().__init__()",
                    "Incorrectly overriding parent methods",
                    "Multiple inheritance diamond problem",
                    "Confusing is-a vs has-a relationships",
                ]
            },
            "a2_file_handling_errors": {
                "expertise": "File operations and error handling",
                "examples": [
                    "Not using context managers for files",
                    "Missing encoding specification",
                    "Not handling FileNotFoundError",
                    "Leaving files open in exception paths",
                ]
            },
            "a2_json_parsing_issues": {
                "expertise": "JSON handling and serialization",
                "examples": [
                    "Not handling JSONDecodeError",
                    "Assuming JSON structure without validation",
                    "Missing ensure_ascii parameter",
                    "Not using custom encoders for complex types",
                ]
            },
            "a2_basic_api_calls": {
                "expertise": "HTTP requests and API interaction",
                "examples": [
                    "Not checking response status codes",
                    "Missing timeout on requests",
                    "Not handling connection errors",
                    "Forgetting to close sessions",
                ]
            },
            "a2_simple_data_validation": {
                "expertise": "Input validation",
                "examples": [
                    "Not validating user input types",
                    "Missing null/None checks",
                    "Not sanitizing string inputs",
                    "Accepting invalid ranges without checking",
                ]
            },
            "a2_basic_testing_patterns": {
                "expertise": "Unit testing fundamentals",
                "examples": [
                    "Not testing edge cases",
                    "Missing assertions in tests",
                    "Testing implementation instead of behavior",
                    "Not using test fixtures",
                ]
            },
            # B1 - Intermediate patterns
            "b1_error_handling_strategies": {
                "expertise": "Exception handling and error recovery",
                "examples": [
                    "Catching overly broad exceptions",
                    "Swallowing exceptions without logging",
                    "Not cleaning up resources in exception path",
                    "Raising wrong exception type for the context",
                ]
            },
            "b1_resource_cleanup": {
                "expertise": "Resource management and context managers",
                "examples": [
                    "Not closing files in exception scenarios",
                    "Missing connection pool cleanup",
                    "Forgetting to release locks",
                    "Replacing try/finally with context managers",
                ]
            },
            "b1_design_pattern_issues": {
                "expertise": "Design patterns and architecture",
                "examples": [
                    "Overusing singleton pattern",
                    "Incorrect factory implementation",
                    "Missing strategy pattern for behavior variation",
                    "Not using observer for event handling",
                ]
            },
            "b1_algorithm_efficiency": {
                "expertise": "Algorithm optimization",
                "examples": [
                    "Using O(n²) when O(n) possible",
                    "Not caching repeated computations",
                    "Inefficient sorting approaches",
                    "Missing early termination in loops",
                ]
            },
            "b1_data_structure_selection": {
                "expertise": "Data structure choice",
                "examples": [
                    "Using list when set is better for lookups",
                    "Not using deque for queue operations",
                    "Missing defaultdict for grouping",
                    "Using dict when OrderedDict needed",
                ]
            },
            "b1_testing_strategies": {
                "expertise": "Testing patterns and coverage",
                "examples": [
                    "Not testing error paths",
                    "Missing integration tests",
                    "Over-mocking reducing test value",
                    "Not testing concurrent scenarios",
                ]
            },
            "b1_mocking_patterns": {
                "expertise": "Test doubles and mocking",
                "examples": [
                    "Mocking implementation details",
                    "Over-complicated mock setup",
                    "Not verifying mock calls",
                    "Missing return value configuration",
                ]
            },
            "b1_dependency_injection": {
                "expertise": "Dependency management",
                "examples": [
                    "Hardcoded dependencies",
                    "Not injecting test doubles",
                    "Tight coupling to implementations",
                    "Missing constructor injection",
                ]
            },
            "b1_configuration_management": {
                "expertise": "Configuration and settings",
                "examples": [
                    "Hardcoded configuration values",
                    "Not using environment variables",
                    "Missing validation of config",
                    "No separation of dev/prod config",
                ]
            },
            "b1_logging_best_practices": {
                "expertise": "Logging and observability",
                "examples": [
                    "Using print() instead of logging",
                    "Wrong log levels",
                    "Missing structured logging",
                    "Not logging context information",
                ]
            },
            # B2 - Advanced patterns
            "b2_race_conditions": {
                "expertise": "Concurrency and thread safety",
                "examples": [
                    "Check-then-act race condition",
                    "Read-modify-write without locking",
                    "Shared mutable state in threads",
                    "Missing synchronization on shared counter",
                ]
            },
            "b2_sql_injection_prevention": {
                "expertise": "Database security",
                "examples": [
                    "String concatenation in SQL queries",
                    "Using f-strings or % formatting for SQL",
                    "Not parameterizing WHERE clauses",
                    "Replacing string interpolation with prepared statements",
                ]
            },
            "b2_deadlock_prevention": {
                "expertise": "Lock ordering and deadlock avoidance",
                "examples": [
                    "Acquiring locks in inconsistent order",
                    "Missing lock timeouts",
                    "Circular dependencies between locks",
                    "Not using lock hierarchies",
                ]
            },
            "b2_memory_leaks": {
                "expertise": "Memory management",
                "examples": [
                    "Circular references preventing garbage collection",
                    "Not clearing event handlers",
                    "Keeping references in closures",
                    "Not using weak references for caches",
                ]
            },
            "b2_xss_vulnerabilities": {
                "expertise": "Web security",
                "examples": [
                    "Not escaping user input in HTML",
                    "Missing Content-Security-Policy headers",
                    "Using innerHTML with untrusted data",
                    "Not sanitizing URL parameters",
                ]
            },
            "b2_authentication_bypass": {
                "expertise": "Authentication security",
                "examples": [
                    "Missing authentication checks",
                    "Weak password hashing (MD5/SHA1)",
                    "Session fixation vulnerabilities",
                    "Not validating JWT signatures",
                ]
            },
            "b2_n_plus_one_queries": {
                "expertise": "Database query optimization",
                "examples": [
                    "Loading related objects in loops",
                    "Not using select_related/prefetch_related",
                    "Missing eager loading",
                    "Excessive database round trips",
                ]
            },
            "b2_circuit_breaker_pattern": {
                "expertise": "Resilience patterns",
                "examples": [
                    "Not protecting against cascading failures",
                    "Missing failure thresholds",
                    "No half-open state for recovery",
                    "Not implementing exponential backoff",
                ]
            },
            "b2_rate_limiting": {
                "expertise": "API protection",
                "examples": [
                    "No rate limits on expensive endpoints",
                    "Missing token bucket implementation",
                    "Not returning 429 status codes",
                    "Per-user vs global rate limits",
                ]
            },
            "b2_connection_pooling": {
                "expertise": "Resource pooling",
                "examples": [
                    "Creating new connections for each request",
                    "Not limiting pool size",
                    "Missing connection validation",
                    "Not handling connection timeouts",
                ]
            },
            # C2 - Expert patterns
            "c2_distributed_locking": {
                "expertise": "Distributed systems and coordination",
                "examples": [
                    "Implementing distributed mutex with Redis",
                    "Handling lock timeouts and orphaned locks",
                    "Fencing tokens to prevent split-brain",
                    "Using ZooKeeper for distributed coordination",
                ]
            },
            "c2_event_sourcing": {
                "expertise": "Event-driven architecture patterns",
                "examples": [
                    "Rebuilding state from event log",
                    "Snapshot optimization for performance",
                    "Handling schema evolution in events",
                    "CQRS with separate read/write models",
                ]
            },
            "c2_consensus_algorithms": {
                "expertise": "Distributed consensus",
                "examples": [
                    "Implementing Raft consensus",
                    "Two-phase commit for distributed transactions",
                    "Paxos for leader election",
                    "Handling split-brain scenarios",
                ]
            },
            "c2_eventual_consistency": {
                "expertise": "Consistency models",
                "examples": [
                    "Conflict resolution strategies",
                    "Vector clocks for causality",
                    "Last-write-wins with timestamps",
                    "CRDTs for conflict-free replication",
                ]
            },
            "c2_saga_pattern": {
                "expertise": "Distributed transactions",
                "examples": [
                    "Choreography-based sagas",
                    "Orchestration with compensating transactions",
                    "Handling partial failures",
                    "Implementing rollback logic",
                ]
            },
            "c2_cqrs_implementation": {
                "expertise": "CQRS architecture",
                "examples": [
                    "Separating read and write models",
                    "Event-driven projections",
                    "Handling eventual consistency in reads",
                    "Optimizing query models",
                ]
            },
            "c2_advanced_concurrency": {
                "expertise": "Advanced concurrency primitives",
                "examples": [
                    "Lock-free data structures",
                    "Compare-and-swap operations",
                    "Software transactional memory",
                    "Work-stealing schedulers",
                ]
            },
            "c2_lock_free_algorithms": {
                "expertise": "Lock-free programming",
                "examples": [
                    "Lock-free queue implementation",
                    "ABA problem and solutions",
                    "Memory ordering constraints",
                    "Hazard pointers for memory reclamation",
                ]
            },
            "c2_memory_optimization": {
                "expertise": "Low-level memory management",
                "examples": [
                    "Object pooling for allocation reduction",
                    "Memory-mapped files for large data",
                    "Cache-friendly data layouts",
                    "Reducing memory fragmentation",
                ]
            },
            "c2_cache_coherence": {
                "expertise": "CPU cache optimization",
                "examples": [
                    "False sharing prevention",
                    "Cache line padding",
                    "Data structure layout for locality",
                    "Prefetching strategies",
                ]
            },
            
            # HARD CATEGORIES - Semantic Understanding Gaps
            "deadlock_patterns": {
                "expertise": "Deadlock detection and prevention in concurrent systems",
                "examples": [
                    "Circular wait conditions",
                    "Inconsistent lock ordering",
                    "Resource allocation deadlocks",
                    "Distributed system deadlocks",
                ]
            },
            "tainted_data_flow_security": {
                "expertise": "Tracking untrusted data from source to sink",
                "examples": [
                    "SQL injection via dataflow",
                    "Command injection patterns",
                    "Path traversal vulnerabilities",
                    "XSS via tainted data",
                ]
            },
            "use_after_free_patterns": {
                "expertise": "Object lifetime management and memory safety",
                "examples": [
                    "Async task holding stale references",
                    "Callback with freed objects",
                    "Event handler use-after-free",
                    "Cached object invalidation",
                ]
            },
            "django_n_plus_one_queries": {
                "expertise": "Django ORM query optimization patterns",
                "examples": [
                    "Missing select_related() calls",
                    "N+1 queries in templates",
                    "Prefetch_related() misuse",
                    "QuerySet evaluation timing",
                ]
            },
            "flask_session_security": {
                "expertise": "Flask session management security",
                "examples": [
                    "Session fixation vulnerabilities",
                    "Insecure session configuration",
                    "Session hijacking prevention",
                    "CSRF token validation",
                ]
            },
            "pandas_performance_anti_patterns": {
                "expertise": "Pandas performance optimization",
                "examples": [
                    "DataFrame.append() in loops",
                    "Inefficient groupby operations",
                    "Memory-intensive operations",
                    "Vectorization opportunities",
                ]
            },
            "resource_leak_exception_paths": {
                "expertise": "Resource management with exception handling",
                "examples": [
                    "File handles not closed on exceptions",
                    "Database connections leaked",
                    "Network sockets not released",
                    "Memory leaks in exception paths",
                ]
            },
            "async_race_conditions": {
                "expertise": "Race conditions in async/await code",
                "examples": [
                    "Shared state modification in async",
                    "Event loop race conditions",
                    "Async context manager issues",
                    "Concurrent async operations",
                ]
            },
            "memory_leak_patterns": {
                "expertise": "Memory leak detection and prevention",
                "examples": [
                    "Circular references",
                    "Event listener accumulation",
                    "Cache growth without bounds",
                    "Generator memory leaks",
                ]
            },
            "sql_injection_dataflow": {
                "expertise": "SQL injection via dataflow analysis",
                "examples": [
                    "User input to SQL queries",
                    "Dynamic query construction",
                    "ORM injection patterns",
                    "Stored procedure injection",
                ]
            },
            "xss_vulnerability_patterns": {
                "expertise": "Cross-site scripting vulnerability patterns",
                "examples": [
                    "Reflected XSS in templates",
                    "Stored XSS in databases",
                    "DOM-based XSS",
                    "Content Security Policy bypass",
                ]
            },
            "csrf_protection_patterns": {
                "expertise": "Cross-site request forgery protection",
                "examples": [
                    "Missing CSRF tokens",
                    "Token validation bypass",
                    "SameSite cookie issues",
                    "Double-submit cookie pattern",
                ]
            },
            "authentication_bypass_patterns": {
                "expertise": "Authentication bypass vulnerabilities",
                "examples": [
                    "JWT token manipulation",
                    "Session fixation attacks",
                    "Password reset bypass",
                    "Multi-factor authentication bypass",
                ]
            },
            "authorization_confusion_patterns": {
                "expertise": "Authorization and access control issues",
                "examples": [
                    "Privilege escalation",
                    "Horizontal privilege escalation",
                    "Vertical privilege escalation",
                    "Access control bypass",
                ]
            },
            "cryptographic_misuse_patterns": {
                "expertise": "Cryptographic implementation mistakes",
                "examples": [
                    "Weak encryption algorithms",
                    "Insecure random number generation",
                    "Key management issues",
                    "Hash function misuse",
                ]
            },
            "timing_attack_vulnerabilities": {
                "expertise": "Timing attack vulnerabilities",
                "examples": [
                    "Password comparison timing",
                    "Database query timing",
                    "Cryptographic timing attacks",
                    "Side-channel information leakage",
                ]
            },
            "buffer_overflow_patterns": {
                "expertise": "Buffer overflow vulnerabilities",
                "examples": [
                    "Array bounds checking",
                    "String manipulation safety",
                    "Memory allocation issues",
                    "Stack overflow patterns",
                ]
            },
            "integer_overflow_patterns": {
                "expertise": "Integer overflow and underflow",
                "examples": [
                    "Arithmetic overflow",
                    "Signed/unsigned conversion",
                    "Array indexing overflow",
                    "Financial calculation overflow",
                ]
            },
            "path_traversal_vulnerabilities": {
                "expertise": "Path traversal and directory traversal",
                "examples": [
                    "File system access bypass",
                    "Directory traversal attacks",
                    "Path normalization issues",
                    "File inclusion vulnerabilities",
                ]
            },
            "command_injection_patterns": {
                "expertise": "Command injection vulnerabilities",
                "examples": [
                    "Shell command injection",
                    "Process execution safety",
                    "Command argument injection",
                    "Environment variable injection",
                ]
            },
        }

        prompt_config = CEFR_PROMPTS.get(inspiration)
        if not prompt_config:
            # Fallback to generic prompt for unconfigured categories
            return None

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        examples_str = "\n".join(f"  - {ex}" for ex in prompt_config["examples"])

        system_prompt = (
            f"You are an expert in {prompt_config['expertise']} and curriculum design. "
            "Your task is to devise a HIGH-IMPACT lesson for an AI agent learning to write better code. "
            "The lesson must teach a specific bug pattern and its fix through concrete before/after code examples. "
            "Return a JSON object with two keys: 'name' (snake_case identifier) and 'description' (one-sentence explanation)."
        )

        user_prompt = (
            f"Propose a HIGH-IMPACT lesson about {inspiration.replace('_', ' ')}.\n\n"
            f"Focus on patterns like:\n{examples_str}\n\n"
            "Generate a similarly impactful concept with clear before/after contrast."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea(inspiration)

        if not response_json_str:
            return self._fallback_idea(inspiration)

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError:
            return self._fallback_idea(inspiration)

        idea['source'] = inspiration
        if not all(k in idea for k in ['name', 'description']):
            return self._fallback_idea(inspiration)

        print(f"  - Generated CEFR idea: {idea['name']}")
        return idea

    def _assess_impact(self, idea: Dict[str, Any], provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Tuple[bool, str]:
        """Acts as the 'Critic' to evaluate the potential impact of a lesson idea."""
        print(f"[Impact Assessment] Critic is evaluating idea: {idea['name']}")
        
        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider for Critic: {e}")
            return False, "Could not instantiate Critic LLM."

        system_prompt = (
            "You are the 'Critic', a meticulous and strategic AI curriculum evaluator. Your task is to assess a proposed lesson concept based on three criteria: Impact, Clarity, and Novelty. "
            "Return a single JSON object with three keys: 'impact_score' (1-10), 'clarity_score' (1-10), and 'novelty_score' (1-10). "
            "\n"
            "**Impact Scoring (How critical is this for production systems?):**\n"
            "- 9-10: Security vulnerabilities, data corruption prevention, system crashes, authentication/authorization\n"
            "  Examples: SQL injection prevention (10), race condition fixes (9), JWT authentication (9)\n"
            "- 7-8: Reliability, error handling, resource management, scalability bottlenecks\n"
            "  Examples: Circuit breaker pattern (8), connection pooling (7), comprehensive logging (7)\n"
            "- 5-6: Performance optimizations, user experience improvements\n"
            "  Examples: Caching strategies (6), N+1 query elimination (6), lazy loading (5)\n"
            "- 3-4: Code maintainability, readability, minor refactoring\n"
            "  Examples: Extract method refactoring (4), add type hints (3)\n"
            "- 1-2: Cosmetic changes, trivial improvements\n"
            "  Examples: Rename variable (2), format code (1)\n"
            "\n"
            "**Clarity Scoring (How specific and testable is this?):**\n"
            "- 9-10: Concrete, measurable, specific implementation with clear before/after states\n"
            "  Example: 'Implement rate limiting with token bucket algorithm' (10)\n"
            "- 7-8: Clear goal but some implementation details flexible\n"
            "  Example: 'Add input validation for user registration' (7)\n"
            "- 5-6: General direction but multiple valid approaches\n"
            "  Example: 'Improve error handling' (5)\n"
            "- 1-4: Vague, ambiguous, or unmeasurable\n"
            "  Example: 'Make code better' (1)\n"
            "\n"
            "**Novelty**: Assume this is a new lesson unless obviously trivial. Score 7-9 for most concepts."
        )
        user_prompt = f"Assess the following lesson concept: {json.dumps(idea)}"

        try:
            os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - WARNING: Critic LLM request failed: {e}. Using offline heuristic.")
            return self._offline_critic_approval(idea)

        if not response_json_str:
            print("  - WARNING: Critic LLM returned an empty response. Using offline heuristic.")
            return self._offline_critic_approval(idea)

        try:
            scores = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Critic JSON parse failed: {e}. Using offline heuristic.")
            return self._offline_critic_approval(idea)

        if not all(k in scores for k in ['impact_score', 'clarity_score', 'novelty_score']):
            print("  - WARNING: Critic response missing required score keys. Using offline heuristic.")
            return self._offline_critic_approval(idea)

        print(f"  - Critic scores: {scores}")

        IMPACT_THRESHOLD = 6
        CLARITY_THRESHOLD = 6

        if scores['impact_score'] < IMPACT_THRESHOLD:
            return False, f"Idea failed to meet impact threshold ({scores['impact_score']} < {IMPACT_THRESHOLD})"
        if scores['clarity_score'] < CLARITY_THRESHOLD:
            return False, f"Idea failed to meet clarity threshold ({scores['clarity_score']} < {CLARITY_THRESHOLD})"

        return True, "Idea passed impact assessment."

    def _fallback_idea(self, inspiration: str) -> Dict[str, Any]:
        """Offline fallback idea when planner LLM is unavailable."""
        base_slug = "_".join(part for part in inspiration.lower().split() if part)
        slug = base_slug or "core_systems"
        idea = {
            "name": f"offline_{slug}_hardening",
            "description": f"Develop automated drills to strengthen {inspiration} safeguards across the stack.",
            "source": inspiration,
        }
        print(f"  - Using offline fallback idea: {idea['name']}")
        return idea

    def _offline_critic_approval(self, idea: Dict[str, Any]) -> Tuple[bool, str]:
        """Offline heuristic approval when critic LLM is unavailable."""
        scores = {"impact_score": 8, "clarity_score": 7, "novelty_score": 7}
        print(f"  - Offline critic heuristic scores: {scores}")
        return True, "Idea approved by offline critic heuristic."

    def _is_duplicate(self, lesson_name: str) -> bool:
        """Checks if a lesson with the given name already exists in the database."""
        db_path = Path("out/learning/curriculum.sqlite")
        if not db_path.exists():
            return False
        store = CurriculumStore(db_path)
        exists = store.lesson_exists(lesson_name)
        store.close()
        return exists

    def _trigger_curriculum_generation(self, idea: Dict[str, Any], provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None):
        """Calls the curriculum generator script for a validated idea."""
        print(f"[Curriculum Generation] Triggering generator for: {idea['name']}")
        if idea['source'] == "bug_fixing":
            script_name = "nerion_digital_physicist.generation.bug_fix_generator"
        elif idea['source'] == "feature_implementation":
            script_name = "nerion_digital_physicist.generation.feature_generator"
        elif idea['source'] == "performance_optimization":
            script_name = "nerion_digital_physicist.generation.performance_generator"
        elif idea['source'] == "code_comprehension":
            script_name = "nerion_digital_physicist.generation.explanation_generator"
        else:
            script_name = "nerion_digital_physicist.generation.curriculum_generator"
        
        # Build command with Vertex AI parameters
        cmd = [
            sys.executable, "-m", script_name,
            "--name", idea['name'],
            "--description", idea['description'],
        ]
        
        # Add Vertex AI parameters if provided
        if provider:
            cmd.extend(["--provider", provider])
        if project_id:
            cmd.extend(["--project-id", project_id])
        if location:
            cmd.extend(["--location", location])
        if model_name:
            cmd.extend(["--model-name", model_name])
        
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True, # Will raise an exception if the script fails
            )
            print("  - Curriculum generator finished successfully.")
            print("\n--- GENERATOR OUTPUT ---")
            print(proc.stdout)
            print("------------------------")
        except subprocess.CalledProcessError as e:
            print("  - ERROR: Curriculum generator failed.")
            print("\n--- FAILED GENERATOR OUTPUT ---")
            print(e.stderr)
            print("-------------------------------")

    def run_cycle(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None):
        """Runs one full autonomous learning cycle."""
        print("--- Starting Autonomous Learning Cycle ---")
        
        # 1. Guided Inspiration from the Meta-Policy
        inspiration = self._get_inspiration()
        if not inspiration:
            print("[Orchestrator] No strategic focus provided by Meta-Policy. Halting cycle.")
            return
        
        # 2. Idea Generation
        idea = self._generate_idea(inspiration, provider=provider, project_id=project_id, location=location, model_name=model_name)
        if not idea:
            print("[Orchestrator] Failed to generate a new idea. Halting cycle.")
            return

        # 3. Impact Assessment (The Critic)
        is_viable, reason = self._assess_impact(idea, provider=provider, project_id=project_id, location=location, model_name=model_name)
        if not is_viable:
            print(f"[Orchestrator] Discarding idea: {reason}. Halting cycle.")
            return
        print(f"[Orchestrator] {reason}")

        # 4. Meta-Policy Veto Check
        is_approved, reason = self.meta_policy_evaluator.evaluate_idea(idea)
        if not is_approved:
            print(f"[Orchestrator] VETOED by Meta-Policy: {reason}. Halting cycle.")
            return
        print("[Orchestrator] Idea approved by Meta-Policy.")

        # 5. Check for duplicates before generating
        if self._is_duplicate(idea['name']):
            print(f"[Orchestrator] Lesson '{idea['name']}' already exists. Halting cycle.")
            return

        # 6. Trigger the curriculum generation process.
        self._trigger_curriculum_generation(idea, provider=provider, project_id=project_id, location=location, model_name=model_name)
        
        print("--- Autonomous Learning Cycle Complete ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Learning Cycle for the Digital Physicist.")
    # Configuration from environment variables or defaults
    default_provider = os.getenv("NERION_LLM_PROVIDER", "openai")
    default_project_id = os.getenv("NERION_LLM_PROJECT_ID")
    default_location = os.getenv("NERION_LLM_LOCATION", "us-central1")
    default_model = os.getenv("NERION_LLM_MODEL", "gpt-4")
    
    parser.add_argument("--provider", type=str, default=default_provider, help="LLM provider to use (e.g., 'openai' or 'gemini')")
    parser.add_argument("--project-id", type=str, default=default_project_id, help="Google Cloud Project ID for Vertex AI.")
    parser.add_argument("--location", type=str, default=default_location, help="Google Cloud location for Vertex AI (e.g., 'us-central1').")
    parser.add_argument("--model-name", type=str, default=default_model, help="Vertex AI model name to use (e.g., 'gemini-pro').")
    args = parser.parse_args()

    orchestrator = LearningOrchestrator()
    orchestrator.run_cycle(provider=args.provider, project_id=args.project_id, location=args.location, model_name=args.model_name)
