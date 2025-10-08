"""
Idea Generator for Autonomous Learning.

This module generates concrete lesson ideas based on inspiration topics using
LLM providers. It routes to specialized generators for different lesson categories
and provides fallback mechanisms when LLMs are unavailable.
"""
import json
from typing import Dict, Any, Optional

from app.parent.coder import Coder


class IdeaGenerator:
    """
    Generates lesson ideas based on inspiration topics.

    Routes to specialized generators for advanced topics (C1/C2 level) and
    uses a general LLM-based generator for CEFR-categorized lessons (A1-B2).
    """

    def __init__(self):
        """Initialize the IdeaGenerator."""
        pass

    def generate(
        self,
        inspiration: str,
        provider: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a lesson idea based on the inspiration source.

        Args:
            inspiration: The inspiration topic to generate a lesson for
            provider: Optional LLM provider override
            project_id: Optional Google Cloud project ID for Vertex AI
            location: Optional Google Cloud location for Vertex AI
            model_name: Optional model name override

        Returns:
            Dictionary with 'name', 'description', and 'source' keys, or None if generation fails
        """
        print(f"[Idea Generation] Generating concept based on: {inspiration}")

        # For CEFR-specific categories, use CEFR generator
        if inspiration.startswith(('a1_', 'a2_', 'b1_', 'b2_', 'c1_', 'c2_')):
            return self._generate_cefr_idea(inspiration, provider, project_id, location, model_name)

        # For all other inspirations, use generic LLM generator
        return self._generate_generic_idea(inspiration, provider, project_id, location, model_name)

    def _generate_generic_idea(
        self,
        inspiration: str,
        provider: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a generic lesson idea using LLM."""
        try:
            llm = Coder(
                role='planner',
                provider_override=provider,
                project_id=project_id,
                location=location,
                model_name=model_name
            )
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert programming educator and curriculum designer. "
            "Your task is to devise a concept for a single, specific, and useful programming lesson for an AI agent. "
            "The lesson concept must be returned as a JSON object with two keys: "
            "'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
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

    def _generate_cefr_idea(
        self,
        inspiration: str,
        provider: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a CEFR-categorized lesson idea.

        Uses enhanced prompts tailored to the specific CEFR level and category.
        """
        # Extract CEFR level and category
        parts = inspiration.split('_', 1)
        if len(parts) < 2:
            return self._generate_generic_idea(inspiration, provider, project_id, location, model_name)

        level = parts[0].upper()  # e.g., "A1", "B2"
        category = parts[1].replace('_', ' ')  # e.g., "variable scope errors"

        try:
            llm = Coder(
                role='planner',
                provider_override=provider,
                project_id=project_id,
                location=location,
                model_name=model_name
            )
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        # Level-specific prompting
        level_guidance = self._get_level_guidance(level)

        system_prompt = (
            f"You are an expert programming educator designing CEFR {level}-level curriculum. "
            f"{level_guidance} "
            "Your task is to devise a specific, practical lesson for an AI agent. "
            "The lesson concept must be returned as a JSON object with two keys: "
            "'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            f"Propose a CEFR {level}-level lesson about: {category}. "
            f"The lesson should be appropriate for {level} difficulty and teach a concrete, practical skill."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request CEFR idea from LLM: {e}")
            return self._fallback_idea(inspiration)

        if not response_json_str:
            print("  - WARNING: CEFR idea generation returned empty. Using offline fallback.")
            return self._fallback_idea(inspiration)

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: CEFR idea JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea(inspiration)

        idea['source'] = inspiration
        idea['cefr_level'] = level

        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: CEFR response missing required keys. Using offline fallback.")
            return self._fallback_idea(inspiration)

        print(f"  - Generated CEFR {level} idea: {idea['name']}")
        return idea

    def _get_level_guidance(self, level: str) -> str:
        """Get prompting guidance for a specific CEFR level."""
        guidance = {
            "A1": "Focus on basic syntax, simple operations, and fundamental concepts. Lessons should be straightforward and introduce one concept at a time.",
            "A2": "Focus on elementary programming patterns, basic OOP, simple data structures, and foundational library usage.",
            "B1": "Focus on intermediate patterns, error handling, testing strategies, and common design patterns.",
            "B2": "Focus on advanced patterns including concurrency, security, performance optimization, and architectural concerns.",
            "C1": "Focus on professional-level skills including distributed systems, advanced algorithms, and production-grade concerns.",
            "C2": "Focus on expert-level topics including consensus algorithms, advanced concurrency patterns, and system-level optimizations."
        }
        return guidance.get(level, "Focus on practical, useful programming skills.")

    def _fallback_idea(self, inspiration: str) -> Dict[str, Any]:
        """
        Generate an offline fallback idea when LLM is unavailable.

        Args:
            inspiration: The inspiration topic

        Returns:
            Dictionary with fallback lesson idea
        """
        base_slug = "_".join(part for part in inspiration.lower().split() if part)
        slug = base_slug or "core_systems"
        idea = {
            "name": f"offline_{slug}_hardening",
            "description": f"Develop automated drills to strengthen {inspiration} safeguards across the stack.",
            "source": inspiration,
        }
        print(f"  - Using offline fallback idea: {idea['name']}")
        return idea
