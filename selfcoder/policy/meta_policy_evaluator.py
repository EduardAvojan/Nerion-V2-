"""
This module reads and interprets the meta_policy.yaml file to provide
high-level strategic guidance to autonomous agents.
"""
from pathlib import Path
import yaml
import random
from typing import Dict, Any, List, Optional, Tuple

CONFIG_PATH = Path("config/meta_policy.yaml")

class MetaPolicyEvaluator:
    """Loads and evaluates the agent's meta-policy."""

    def __init__(self):
        self._policy = self._load_policy()

    def _load_policy(self) -> Dict[str, Any]:
        """Loads the meta_policy.yaml file."""
        if not CONFIG_PATH.exists():
            print(f"[MetaPolicy] WARNING: Policy file not found at {CONFIG_PATH}. Using empty policy.")
            return {}
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[MetaPolicy] ERROR: Could not parse policy file: {e}")
            return {}

    def get_strategic_focus(self) -> Optional[str]:
        """Returns a focus area based on the current strategic directives."""
        directives = self._policy.get("strategic_directives", {})
        focus_areas = directives.get("CURRENT_FOCUS", [])
        
        if not focus_areas:
            print("[MetaPolicy] No strategic focus areas defined. Inspiration will be unguided.")
            return None
        
        # Pick one of the defined focus areas at random to guide inspiration.
        return random.choice(focus_areas)

    def evaluate_idea(self, idea: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluates a generated lesson idea against the meta-policy.
        
        Returns (is_approved, reason).
        """
        directives = self._policy.get("strategic_directives", {})
        avoid_topics = directives.get("AVOID_TOPICS", [])

        # Check if the idea's description contains any forbidden topics.
        description = idea.get("description", "").lower()
        for topic in avoid_topics:
            if topic.lower() in description:
                reason = f"Idea rejected: Description contains forbidden topic '{topic}'."
                print(f"[MetaPolicy] {reason}")
                return False, reason

        # Placeholder for more advanced ethical guardrail checks.
        
        return True, "Idea approved by Meta-Policy."
