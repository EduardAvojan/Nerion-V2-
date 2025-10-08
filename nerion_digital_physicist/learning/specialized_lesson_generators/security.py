"""
C1-Level Specialized Lesson Generators - Security.

This module contains specialized generator methods for security hardening and data validation.
"""
import json
from typing import Dict, Any, Optional

from app.parent.coder import Coder
from .cefr_prompts import CEFR_PROMPTS


class SecurityGenerators:
    """Generators for Security."""

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

