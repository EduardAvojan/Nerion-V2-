"""
C1-Level Specialized Lesson Generators - General.

This module contains specialized generator methods for general bug fixing, features, and code comprehension.
"""
import json
from typing import Dict, Any, Optional

from app.parent.coder import Coder
from .cefr_prompts import CEFR_PROMPTS


class GeneralGenerators:
    """Generators for General."""

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

