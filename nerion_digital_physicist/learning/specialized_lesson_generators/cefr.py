"""
CEFR-Level Lesson Generator.

This module contains the generator for CEFR-categorized lesson inspirations.
"""
import json
from typing import Dict, Any, Optional

from app.parent.coder import Coder
from .cefr_prompts import CEFR_PROMPTS


class CEFRGenerator:
    """Generator for CEFR-categorized lessons."""

    def _fallback_idea(self, inspiration: str) -> Dict[str, Any]:
        """Generate an offline fallback idea when LLM is unavailable."""
        base_slug = "_".join(part for part in inspiration.lower().split() if part)
        slug = base_slug or "core_systems"
        idea = {
            "name": f"offline_{slug}_hardening",
            "description": f"Develop automated drills to strengthen {inspiration} safeguards across the stack.",
            "source": inspiration,
        }
        print(f"  - Using offline fallback idea: {idea['name']}")
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
