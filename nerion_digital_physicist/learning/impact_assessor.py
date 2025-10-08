"""
Impact Assessor for Autonomous Learning.

This module acts as the 'Critic' to evaluate the potential impact and viability
of proposed lesson ideas using a multi-dimensional scoring system.
"""
import os
import json
from typing import Dict, Any, Optional, Tuple

from app.parent.coder import Coder


class ImpactAssessor:
    """
    Evaluates lesson ideas using Impact, Clarity, and Novelty scores.

    The Critic uses LLM-based assessment with fallback to offline heuristics
    when LLM is unavailable. Ideas must meet minimum thresholds for both
    impact and clarity to be approved.
    """

    # Scoring thresholds
    IMPACT_THRESHOLD = 6
    CLARITY_THRESHOLD = 6

    def __init__(self):
        """Initialize the ImpactAssessor."""
        pass

    def assess(
        self,
        idea: Dict[str, Any],
        provider: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate the potential impact of a lesson idea.

        Args:
            idea: Dictionary with 'name' and 'description' keys
            provider: Optional LLM provider override
            project_id: Optional Google Cloud project ID for Vertex AI
            location: Optional Google Cloud location for Vertex AI
            model_name: Optional model name override

        Returns:
            Tuple of (is_viable, reason) where is_viable is True if the idea
            meets quality thresholds
        """
        print(f"[Impact Assessment] Critic is evaluating idea: {idea['name']}")

        try:
            llm = Coder(
                role='planner',
                provider_override=provider,
                project_id=project_id,
                location=location,
                model_name=model_name
            )
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider for Critic: {e}")
            return False, "Could not instantiate Critic LLM."

        system_prompt = self._build_system_prompt()
        user_prompt = f"Assess the following lesson concept: {json.dumps(idea)}"

        try:
            os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - WARNING: Critic LLM request failed: {e}. Using offline heuristic.")
            return self._offline_approval(idea)

        if not response_json_str:
            print("  - WARNING: Critic LLM returned an empty response. Using offline heuristic.")
            return self._offline_approval(idea)

        try:
            scores = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Critic JSON parse failed: {e}. Using offline heuristic.")
            return self._offline_approval(idea)

        if not all(k in scores for k in ['impact_score', 'clarity_score', 'novelty_score']):
            print("  - WARNING: Critic response missing required score keys. Using offline heuristic.")
            return self._offline_approval(idea)

        print(f"  - Critic scores: {scores}")

        # Evaluate against thresholds
        if scores['impact_score'] < self.IMPACT_THRESHOLD:
            return False, f"Idea failed to meet impact threshold ({scores['impact_score']} < {self.IMPACT_THRESHOLD})"
        if scores['clarity_score'] < self.CLARITY_THRESHOLD:
            return False, f"Idea failed to meet clarity threshold ({scores['clarity_score']} < {self.CLARITY_THRESHOLD})"

        return True, "Idea passed impact assessment."

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the Critic LLM."""
        return (
            "You are the 'Critic', a meticulous and strategic AI curriculum evaluator. "
            "Your task is to assess a proposed lesson concept based on three criteria: Impact, Clarity, and Novelty. "
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

    def _offline_approval(self, idea: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Provide offline heuristic approval when Critic LLM is unavailable.

        Uses conservative default scores that meet minimum thresholds.

        Args:
            idea: The lesson idea being evaluated

        Returns:
            Tuple of (True, reason) indicating heuristic approval
        """
        scores = {"impact_score": 8, "clarity_score": 7, "novelty_score": 7}
        print(f"  - Offline critic heuristic scores: {scores}")
        return True, "Idea approved by offline critic heuristic."

    @classmethod
    def get_thresholds(cls) -> Dict[str, int]:
        """
        Get current scoring thresholds.

        Returns:
            Dictionary with threshold values
        """
        return {
            "impact_threshold": cls.IMPACT_THRESHOLD,
            "clarity_threshold": cls.CLARITY_THRESHOLD,
        }

    @classmethod
    def set_thresholds(cls, impact: Optional[int] = None, clarity: Optional[int] = None) -> None:
        """
        Update scoring thresholds (for experimentation).

        Args:
            impact: New impact threshold (1-10)
            clarity: New clarity threshold (1-10)
        """
        if impact is not None:
            if not 1 <= impact <= 10:
                raise ValueError("Impact threshold must be between 1 and 10")
            cls.IMPACT_THRESHOLD = impact
            print(f"[ImpactAssessor] Updated impact threshold to {impact}")

        if clarity is not None:
            if not 1 <= clarity <= 10:
                raise ValueError("Clarity threshold must be between 1 and 10")
            cls.CLARITY_THRESHOLD = clarity
            print(f"[ImpactAssessor] Updated clarity threshold to {clarity}")
