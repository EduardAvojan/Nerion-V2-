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
from typing import Dict, Any, Optional, Tuple

from app.parent.coder import Coder
from selfcoder.policy.meta_policy_evaluator import MetaPolicyEvaluator


class LearningOrchestrator:
    """Manages the autonomous learning cycle for the Digital Physicist."""

    def __init__(self):
        self._knowledge_base = set()
        self.meta_policy_evaluator = MetaPolicyEvaluator()

    def _get_inspiration(self) -> Optional[str]:
        """Selects a source of inspiration based on the meta-policy."""
        focus = self.meta_policy_evaluator.get_strategic_focus()
        if focus:
            print(f"[Inspiration] Meta-Policy selected strategic focus: {focus}")
        return focus

    def _generate_idea(self, inspiration: str) -> Optional[Dict[str, Any]]:
        """Generates a specific lesson idea based on the inspiration source."""
        print(f"[Idea Generation] Generating concept based on: {inspiration}")
        
        try:
            llm = Coder(role='planner')
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
            if not response_json_str:
                print("  - ERROR: Idea generation LLM returned an empty response.")
                return None
            
            idea = json.loads(response_json_str)
            idea['source'] = inspiration # Add the source for context
            
            if not all(k in idea for k in ['name', 'description']):
                print("  - ERROR: LLM response was missing required keys ('name', 'description').")
                return None

            print(f"  - Generated idea: {idea['name']}")
            return idea
        except Exception as e:
            print(f"  - ERROR: Failed to generate or parse LLM response for idea generation: {e}")
            return None

    def _assess_impact(self, idea: Dict[str, Any]) -> Tuple[bool, str]:
        """Acts as the 'Critic' to evaluate the potential impact of a lesson idea."""
        print(f"[Impact Assessment] Critic is evaluating idea: {idea['name']}")
        
        try:
            llm = Coder(role='planner')
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider for Critic: {e}")
            return False, "Could not instantiate Critic LLM."

        system_prompt = (
            "You are the 'Critic', a meticulous and strategic AI curriculum evaluator. Your task is to assess a proposed lesson concept based on three criteria: Impact, Clarity, and Novelty. "
            "Return a single JSON object with three keys: 'impact_score' (1-10), 'clarity_score' (1-10), and 'novelty_score' (1-10). "
            "- **Impact**: How critical is this skill? (Security/Stability=9-10, Reliability=7-8, Performance=5-6, Maintainability/Style=2-4). "
            "- **Clarity**: Is the concept specific, testable, and unambiguous? 'Implement exponential backoff' is a 9; 'make code better' is a 1. "
            "- **Novelty**: How new is this concept? (Assume for now this is the first lesson, so score it high)."
        )
        user_prompt = f"Assess the following lesson concept: {json.dumps(idea)}"

        try:
            os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
            if not response_json_str:
                return False, "Critic LLM returned an empty response."
            
            scores = json.loads(response_json_str)
            
            if not all(k in scores for k in ['impact_score', 'clarity_score', 'novelty_score']):
                return False, "Critic LLM response was missing required score keys."

            print(f"  - Critic scores: {scores}")

            IMPACT_THRESHOLD = 7
            CLARITY_THRESHOLD = 6

            if scores['impact_score'] < IMPACT_THRESHOLD:
                return False, f"Idea failed to meet impact threshold ({scores['impact_score']} < {IMPACT_THRESHOLD})"
            if scores['clarity_score'] < CLARITY_THRESHOLD:
                return False, f"Idea failed to meet clarity threshold ({scores['clarity_score']} < {CLARITY_THRESHOLD})"
            
            return True, "Idea passed impact assessment."

        except Exception as e:
            return False, f"Failed to generate or parse Critic LLM response: {e}"

    def _trigger_curriculum_generation(self, idea: Dict[str, Any]):
        """Calls the curriculum generator script for a validated idea."""
        print(f"[Curriculum Generation] Triggering generator for: {idea['name']}")
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "nerion_digital_physicist.generation.curriculum_generator",
                    "--name", idea['name'],
                    "--description", idea['description'],
                ],
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

    def run_cycle(self):
        """Runs one full autonomous learning cycle."""
        print("--- Starting Autonomous Learning Cycle ---")
        
        # 1. Guided Inspiration from the Meta-Policy
        inspiration = self._get_inspiration()
        if not inspiration:
            print("[Orchestrator] No strategic focus provided by Meta-Policy. Halting cycle.")
            return
        
        # 2. Idea Generation
        idea = self._generate_idea(inspiration)
        if not idea:
            print("[Orchestrator] Failed to generate a new idea. Halting cycle.")
            return

        # 3. Impact Assessment (The Critic)
        is_viable, reason = self._assess_impact(idea)
        if not is_viable:
            print(f"[Orchestrator] Discarding idea: {reason}. Halting cycle.")
            return
        print(f"[Orchestrator] {reason}")

        # 4. Meta-Policy Veto Check
        is_approved, reason = self.meta_policy_evaluator.evaluate_idea(idea)
        if not is_approved:
            print(f"[Orchestrator] VETOED by Meta-Policy: {reason}. Halting cycle.")
            return
        print(f"[Orchestrator] Idea approved by Meta-Policy.")

        # 5. Trigger the curriculum generation process.
        self._trigger_curriculum_generation(idea)
        
        print("--- Autonomous Learning Cycle Complete ---")


if __name__ == "__main__":
    orchestrator = LearningOrchestrator()
    orchestrator.run_cycle()