"""
This module contains the Learning Orchestrator, which implements the
Autonomous Impact Framework for self-directed learning.

It is responsible for:
1.  Finding inspiration for new lesson topics.
2.  Generating concrete lesson ideas.
3.  Assessing the impact and viability of those ideas.
4.  Triggering the curriculum generator for high-quality ideas.

This module now delegates to specialized components in the learning/ package
for better separation of concerns and maintainability.
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

from nerion_digital_physicist.learning import (
    InspirationSelector,
    IdeaGenerator,
    ImpactAssessor,
    LessonValidator,
    SpecializedLessonGenerators,
)


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
    """
    Autonomous learning orchestrator for the Digital Physicist system.

    This class coordinates the self-directed learning cycle by delegating to
    specialized components:
    - InspirationSelector: Chooses lesson topics
    - IdeaGenerator: Generates concrete lesson ideas
    - ImpactAssessor: Evaluates lesson viability
    - LessonValidator: Validates against duplicates and policy

    Example:
        >>> orchestrator = LearningOrchestrator()
        >>> orchestrator.run_cycle(provider="openai", model_name="gpt-4")
    """

    def __init__(self, category_filter: Optional[str] = None) -> None:
        """
        Initialize the LearningOrchestrator.

        Args:
            category_filter: Optional category to filter lesson generation to
        """
        self.inspiration_selector = InspirationSelector(category_filter=category_filter)
        self.idea_generator = IdeaGenerator()
        self.impact_assessor = ImpactAssessor()
        self.lesson_validator = LessonValidator()
        self.specialized_generators = SpecializedLessonGenerators()  # Specialized generators for all CEFR levels

    def _get_inspiration(self) -> Optional[str]:
        """
        Select a source of inspiration based on the meta-policy.

        Delegates to InspirationSelector for topic selection.

        Returns:
            Selected inspiration topic, or None if selection fails
        """
        return self.inspiration_selector.select()

    def _generate_idea(self, inspiration: str, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """
        Generate a specific lesson idea based on the inspiration source.

        Routes to SpecializedLessonGenerators for domain-specific topics, otherwise
        delegates to IdeaGenerator for CEFR-based and generic lessons.
        """
        # Route to specialized generators (organized by domain)
        specialized_generator_map = {
            "refactoring": self.specialized_generators._generate_refactoring_idea,
            "bug_fixing": self.specialized_generators._generate_bug_fix_idea,
            "feature_implementation": self.specialized_generators._generate_feature_implementation_idea,
            "performance_optimization": self.specialized_generators._generate_performance_optimization_idea,
            "code_comprehension": self.specialized_generators._generate_code_comprehension_idea,
            "security_hardening": self.specialized_generators._generate_security_idea,
            "testing_strategies": self.specialized_generators._generate_testing_idea,
            "api_design": self.specialized_generators._generate_api_design_idea,
            "concurrency_patterns": self.specialized_generators._generate_concurrency_idea,
            "data_structures": self.specialized_generators._generate_data_structure_idea,
            "algorithmic_optimization": self.specialized_generators._generate_algorithm_idea,
            "error_recovery": self.specialized_generators._generate_error_recovery_idea,
            "database_design": self.specialized_generators._generate_database_idea,
            "observability": self.specialized_generators._generate_observability_idea,
            "deployment_cicd": self.specialized_generators._generate_deployment_idea,
            "caching_strategies": self.specialized_generators._generate_caching_idea,
            "message_queues": self.specialized_generators._generate_message_queue_idea,
            "resource_management": self.specialized_generators._generate_resource_management_idea,
            "distributed_systems": self.specialized_generators._generate_distributed_systems_idea,
            "scaling_patterns": self.specialized_generators._generate_scaling_idea,
            "data_validation": self.specialized_generators._generate_data_validation_idea,
            "configuration_management": self.specialized_generators._generate_configuration_idea,
        }

        if inspiration in specialized_generator_map:
            return specialized_generator_map[inspiration](provider, project_id, location, model_name)

        # Also check if it's a CEFR idea that needs the CEFR generator
        if inspiration.startswith(('a1_', 'a2_', 'b1_', 'b2_', 'c1_', 'c2_')) or inspiration in SpecializedLessonGenerators._generate_cefr_idea.__code__.co_names:
            # Use SpecializedLessonGenerators CEFR generator for categorized inspirations
            cefr_idea = self.specialized_generators._generate_cefr_idea(inspiration, provider, project_id, location, model_name)
            if cefr_idea:
                return cefr_idea

        # For all other inspirations (generic), delegate to IdeaGenerator
        return self.idea_generator.generate(
            inspiration,
            provider=provider,
            project_id=project_id,
            location=location,
            model_name=model_name
        )

    def _assess_impact(self, idea: Dict[str, Any], provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None):
        """
        Evaluate the potential impact of a lesson idea.

        Delegates to ImpactAssessor for scoring and threshold evaluation.
        """
        return self.impact_assessor.assess(
            idea,
            provider=provider,
            project_id=project_id,
            location=location,
            model_name=model_name
        )

    def _is_duplicate(self, lesson_name: str) -> bool:
        """
        Check if a lesson with the given name already exists.

        Delegates to LessonValidator for duplicate checking.
        """
        return self.lesson_validator.is_duplicate(lesson_name)

    def _trigger_curriculum_generation(self, idea: Dict[str, Any], provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None):
        """Calls the curriculum generator script for a validated idea."""
        print(f"[Curriculum Generation] Triggering generator for: {idea['name']}")
        if idea['source'] == "bug_fixing":
            script_name = "nerion_digital_physicist.generation.bug_fix_generator"
        elif idea['source'] == "feature_implementation":
            script_name = "nerion_digital_physicist.generation.feature_generator"
        elif idea['source'] == "performance_optimization":
            script_name = "nerion_digital_physicist.generation.performance_generator"
        elif idea['source'] == "code_comprehension":
            script_name = "nerion_digital_physicist.generation.explanation_generator"
        else:
            script_name = "nerion_digital_physicist.generation.curriculum_generator"
        
        # Build command with Vertex AI parameters
        cmd = [
            sys.executable, "-m", script_name,
            "--name", idea['name'],
            "--description", idea['description'],
        ]
        
        # Add Vertex AI parameters if provided
        if provider:
            cmd.extend(["--provider", provider])
        if project_id:
            cmd.extend(["--project-id", project_id])
        if location:
            cmd.extend(["--location", location])
        if model_name:
            cmd.extend(["--model-name", model_name])
        
        try:
            proc = subprocess.run(
                cmd,
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

    def run_cycle(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None):
        """
        Run one full autonomous learning cycle.

        This method coordinates the entire learning workflow:
        1. Select inspiration topic
        2. Generate lesson idea
        3. Assess impact/viability
        4. Validate against duplicates and policy
        5. Trigger curriculum generation

        Args:
            provider: Optional LLM provider override
            project_id: Optional Google Cloud project ID
            location: Optional Google Cloud location
            model_name: Optional model name override
        """
        print("--- Starting Autonomous Learning Cycle ---")

        # 1. Guided Inspiration from the Meta-Policy
        inspiration = self._get_inspiration()
        if not inspiration:
            print("[Orchestrator] No strategic focus provided by Meta-Policy. Halting cycle.")
            return

        # 2. Idea Generation
        idea = self._generate_idea(inspiration, provider=provider, project_id=project_id, location=location, model_name=model_name)
        if not idea:
            print("[Orchestrator] Failed to generate a new idea. Halting cycle.")
            return

        # 3. Impact Assessment (The Critic)
        is_viable, reason = self._assess_impact(idea, provider=provider, project_id=project_id, location=location, model_name=model_name)
        if not is_viable:
            print(f"[Orchestrator] Discarding idea: {reason}. Halting cycle.")
            return
        print(f"[Orchestrator] {reason}")

        # 4. Validation (duplicates + meta-policy)
        is_valid, validation_reason = self.lesson_validator.validate(idea)
        if not is_valid:
            print(f"[Orchestrator] {validation_reason}. Halting cycle.")
            return
        print("[Orchestrator] Lesson validated successfully.")

        # 5. Trigger the curriculum generation process
        self._trigger_curriculum_generation(idea, provider=provider, project_id=project_id, location=location, model_name=model_name)

        print("--- Autonomous Learning Cycle Complete ---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Learning Cycle for the Digital Physicist.")
    # Configuration from environment variables or defaults
    default_provider = os.getenv("NERION_LLM_PROVIDER", "openai")
    default_project_id = os.getenv("NERION_LLM_PROJECT_ID")
    default_location = os.getenv("NERION_LLM_LOCATION", "us-central1")
    default_model = os.getenv("NERION_LLM_MODEL", "gpt-4")
    
    parser.add_argument("--provider", type=str, default=default_provider, help="LLM provider to use (e.g., 'openai' or 'gemini')")
    parser.add_argument("--project-id", type=str, default=default_project_id, help="Google Cloud Project ID for Vertex AI.")
    parser.add_argument("--location", type=str, default=default_location, help="Google Cloud location for Vertex AI (e.g., 'us-central1').")
    parser.add_argument("--model-name", type=str, default=default_model, help="Vertex AI model name to use (e.g., 'gemini-pro').")
    args = parser.parse_args()

    orchestrator = LearningOrchestrator()
    orchestrator.run_cycle(provider=args.provider, project_id=args.project_id, location=args.location, model_name=args.model_name)
