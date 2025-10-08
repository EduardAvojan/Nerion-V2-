"""
C1-Level Specialized Lesson Generators - Code Quality.

This module contains specialized generator methods for code quality, refactoring, testing, and API design.
"""
import json
from typing import Dict, Any, Optional

from app.parent.coder import Coder
from .cefr_prompts import CEFR_PROMPTS


class CodeQualityGenerators:
    """Generators for Code Quality."""

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

