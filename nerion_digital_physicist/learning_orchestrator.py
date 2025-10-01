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
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

from app.parent.coder import Coder
from selfcoder.policy.meta_policy_evaluator import MetaPolicyEvaluator
from nerion_digital_physicist.db.curriculum_store import CurriculumStore


def _load_environment() -> None:
    """Load environment variables from .env when available."""
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    except Exception:
        # Silent failure keeps behaviour consistent with optional dotenv import.
        pass


_load_environment()


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

    def _generate_idea(self, inspiration: str, provider: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a specific lesson idea based on the inspiration source."""
        print(f"[Idea Generation] Generating concept based on: {inspiration}")
        
        try:
            llm = Coder(role='planner', provider_override=provider)
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

    def _assess_impact(self, idea: Dict[str, Any], provider: str | None = None) -> Tuple[bool, str]:
        """Acts as the 'Critic' to evaluate the potential impact of a lesson idea."""
        print(f"[Impact Assessment] Critic is evaluating idea: {idea['name']}")
        
        try:
            llm = Coder(role='planner', provider_override=provider)
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
        except Exception as e:
            print(f"  - WARNING: Critic LLM request failed: {e}. Using offline heuristic.")
            return self._offline_critic_approval(idea)

        if not response_json_str:
            print("  - WARNING: Critic LLM returned an empty response. Using offline heuristic.")
            return self._offline_critic_approval(idea)

        try:
            scores = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Critic JSON parse failed: {e}. Using offline heuristic.")
            return self._offline_critic_approval(idea)

        if not all(k in scores for k in ['impact_score', 'clarity_score', 'novelty_score']):
            print("  - WARNING: Critic response missing required score keys. Using offline heuristic.")
            return self._offline_critic_approval(idea)

        print(f"  - Critic scores: {scores}")

        IMPACT_THRESHOLD = 7
        CLARITY_THRESHOLD = 6

        if scores['impact_score'] < IMPACT_THRESHOLD:
            return False, f"Idea failed to meet impact threshold ({scores['impact_score']} < {IMPACT_THRESHOLD})"
        if scores['clarity_score'] < CLARITY_THRESHOLD:
            return False, f"Idea failed to meet clarity threshold ({scores['clarity_score']} < {CLARITY_THRESHOLD})"

        return True, "Idea passed impact assessment."

    def _fallback_idea(self, inspiration: str) -> Dict[str, Any]:
        """Offline fallback idea when planner LLM is unavailable."""
        base_slug = "_".join(part for part in inspiration.lower().split() if part)
        slug = base_slug or "core_systems"
        idea = {
            "name": f"offline_{slug}_hardening",
            "description": f"Develop automated drills to strengthen {inspiration} safeguards across the stack.",
            "source": inspiration,
        }
        print(f"  - Using offline fallback idea: {idea['name']}")
        return idea

    def _offline_critic_approval(self, idea: Dict[str, Any]) -> Tuple[bool, str]:
        """Offline heuristic approval when critic LLM is unavailable."""
        scores = {"impact_score": 8, "clarity_score": 7, "novelty_score": 7}
        print(f"  - Offline critic heuristic scores: {scores}")
        return True, "Idea approved by offline critic heuristic."

    def _is_duplicate(self, lesson_name: str) -> bool:
        """Checks if a lesson with the given name already exists in the database."""
        db_path = Path("out/learning/curriculum.sqlite")
        if not db_path.exists():
            return False
        store = CurriculumStore(db_path)
        exists = store.lesson_exists(lesson_name)
        store.close()
        return exists

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

    def run_cycle(self, provider: str | None = None):
        """Runs one full autonomous learning cycle."""
        print("--- Starting Autonomous Learning Cycle ---")
        
        # 1. Guided Inspiration from the Meta-Policy
        inspiration = self._get_inspiration()
        if not inspiration:
            print("[Orchestrator] No strategic focus provided by Meta-Policy. Halting cycle.")
            return
        
        # 2. Idea Generation
        idea = self._generate_idea(inspiration, provider=provider)
        if not idea:
            print("[Orchestrator] Failed to generate a new idea. Halting cycle.")
            return

        # 3. Impact Assessment (The Critic)
        is_viable, reason = self._assess_impact(idea, provider=provider)
        if not is_viable:
            print(f"[Orchestrator] Discarding idea: {reason}. Halting cycle.")
            return
        print(f"[Orchestrator] {reason}")

        # 4. Meta-Policy Veto Check
        is_approved, reason = self.meta_policy_evaluator.evaluate_idea(idea)
        if not is_approved:
            print(f"[Orchestrator] VETOED by Meta-Policy: {reason}. Halting cycle.")
            return
        print("[Orchestrator] Idea approved by Meta-Policy.")

        # 5. Check for duplicates before generating
        if self._is_duplicate(idea['name']):
            print(f"[Orchestrator] Lesson '{idea['name']}' already exists. Halting cycle.")
            return

        # 6. Trigger the curriculum generation process.
        self._trigger_curriculum_generation(idea)
        
        print("--- Autonomous Learning Cycle Complete ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Learning Cycle for the Digital Physicist.")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider to use (e.g., 'openai' or 'gemini')")
    args = parser.parse_args()

    orchestrator = LearningOrchestrator()
    orchestrator.run_cycle(provider=args.provider)
