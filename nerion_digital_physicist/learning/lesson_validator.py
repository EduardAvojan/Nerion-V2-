"""
Lesson Validator for Autonomous Learning.

This module validates lesson ideas for duplicates and policy compliance before
triggering curriculum generation.
"""
from pathlib import Path
from typing import Dict, Any, Tuple

from selfcoder.policy.meta_policy_evaluator import MetaPolicyEvaluator
from nerion_digital_physicist.db.curriculum_store import CurriculumStore


class LessonValidator:
    """
    Validates lesson ideas for duplicates and policy compliance.

    Performs two main checks:
    1. Duplicate check: Ensures the lesson doesn't already exist
    2. Policy check: Verifies the lesson complies with meta-policy rules
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize the LessonValidator.

        Args:
            db_path: Optional path to curriculum database. If None, uses default location.
        """
        self.db_path = db_path or Path("out/learning/curriculum.sqlite")
        self.meta_policy_evaluator = MetaPolicyEvaluator()

    def validate(self, idea: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a lesson idea for duplicates and policy compliance.

        Args:
            idea: Dictionary with 'name' and 'description' keys

        Returns:
            Tuple of (is_valid, reason) where is_valid is True if the lesson
            passes all validation checks
        """
        # Check for duplicates
        if self.is_duplicate(idea['name']):
            return False, f"Lesson '{idea['name']}' already exists in database"

        # Check meta-policy approval
        is_approved, reason = self.meta_policy_evaluator.evaluate_idea(idea)
        if not is_approved:
            return False, f"VETOED by Meta-Policy: {reason}"

        return True, "Lesson validated successfully"

    def is_duplicate(self, lesson_name: str) -> bool:
        """
        Check if a lesson with the given name already exists in the database.

        Args:
            lesson_name: Name of the lesson to check

        Returns:
            True if lesson exists, False otherwise
        """
        if not self.db_path.exists():
            return False

        store = CurriculumStore(self.db_path)
        try:
            exists = store.lesson_exists(lesson_name)
            return exists
        finally:
            store.close()

    def check_meta_policy(self, idea: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if the lesson idea complies with meta-policy rules.

        Args:
            idea: Dictionary with lesson idea details

        Returns:
            Tuple of (is_approved, reason)
        """
        return self.meta_policy_evaluator.evaluate_idea(idea)

    def get_existing_lesson_count(self) -> int:
        """
        Get the total number of lessons in the database.

        Returns:
            Count of existing lessons
        """
        if not self.db_path.exists():
            return 0

        store = CurriculumStore(self.db_path)
        try:
            # Get all lessons and count them
            lessons = store.get_all_lessons()
            return len(lessons) if lessons else 0
        finally:
            store.close()

    def get_lessons_by_category(self, category: str) -> list[str]:
        """
        Get all lesson names for a specific category.

        Args:
            category: Category prefix (e.g., "a1", "b2", "refactoring")

        Returns:
            List of lesson names in that category
        """
        if not self.db_path.exists():
            return []

        store = CurriculumStore(self.db_path)
        try:
            lessons = store.get_all_lessons()
            if not lessons:
                return []

            # Filter by category prefix
            category_lower = category.lower()
            return [
                lesson['name'] for lesson in lessons
                if lesson['name'].startswith(category_lower)
            ]
        finally:
            store.close()
