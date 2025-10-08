"""
C1-Level Specialized Lesson Generators - Systems.

This module contains specialized generator methods for distributed systems, scaling, concurrency, and databases.
"""
import json
from typing import Dict, Any, Optional

from app.parent.coder import Coder
from .cefr_prompts import CEFR_PROMPTS


class SystemsGenerators:
    """Generators for Systems."""

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

