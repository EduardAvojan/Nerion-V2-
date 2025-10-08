"""
C1-Level Specialized Lesson Generators - Performance.

This module contains specialized generator methods for performance optimization, caching, and algorithms.
"""
import json
from typing import Dict, Any, Optional

from app.parent.coder import Coder
from .cefr_prompts import CEFR_PROMPTS


class PerformanceGenerators:
    """Generators for Performance."""

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

