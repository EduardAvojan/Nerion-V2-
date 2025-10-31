"""
Auto-Curriculum Generator

Automatically generates curriculum lessons from production bugs.

Workflow:
1. Sample high-priority bugs from ReplayStore
2. Synthesize lesson from bug (before_code, after_code, test)
3. Validate using existing LessonValidator (no duplicates)
4. Store in curriculum.sqlite

Integration points:
- ReplayStore: Source of production bugs
- LessonValidator: Reuse from LearningOrchestrator
- SafeCurriculumDB: Store lessons with duplicate protection
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import existing infrastructure
from nerion_digital_physicist.infrastructure.memory import ReplayStore, Experience
from nerion_digital_physicist.learning import LessonValidator


@dataclass
class SynthesizedLesson:
    """A lesson synthesized from production bug"""
    name: str
    description: str
    before_code: str                 # Buggy code
    after_code: str                  # Fixed code
    test_code: str                   # Test that catches bug
    language: str
    difficulty: str                  # A1-C2 CERF level
    category: str                    # Bug type
    tags: List[str]
    provenance: str                  # "production_bug"
    source_bug_id: str
    surprise_score: float


@dataclass
class GenerationMetrics:
    """Metrics for auto-curriculum generation"""
    bugs_processed: int = 0
    lessons_generated: int = 0
    lessons_rejected: int = 0       # Duplicates or invalid
    by_bug_type: Dict[str, int] = None
    by_difficulty: Dict[str, int] = None
    avg_surprise: float = 0.0
    last_generation_time: Optional[str] = None

    def __post_init__(self):
        if self.by_bug_type is None:
            self.by_bug_type = {}
        if self.by_difficulty is None:
            self.by_difficulty = {}


class AutoCurriculumGenerator:
    """
    Automatically generates curriculum lessons from production bugs.

    Usage:
        >>> replay_store = ReplayStore(Path("data/replay"))
        >>> validator = LessonValidator()
        >>> generator = AutoCurriculumGenerator(replay_store, validator)
        >>>
        >>> # Generate lessons from high-priority bugs
        >>> lessons = generator.generate_from_production(k=50)
        >>>
        >>> # Store in curriculum
        >>> from nerion_digital_physicist.db.safe_curriculum import SafeCurriculumDB
        >>> db = SafeCurriculumDB("out/learning/curriculum.sqlite")
        >>> for lesson in lessons:
        ...     db.add_lesson(lesson.__dict__)
    """

    def __init__(
        self,
        replay_store: ReplayStore,
        validator: LessonValidator,
        llm_provider: str = "gemini"
    ):
        """
        Initialize auto-curriculum generator.

        Args:
            replay_store: ReplayStore with production bugs
            validator: LessonValidator for duplicate checking
            llm_provider: LLM provider for lesson synthesis
        """
        self.replay_store = replay_store
        self.validator = validator
        self.llm_provider = llm_provider
        self.metrics = GenerationMetrics()

        # Initialize LLM client
        self.llm_client = self._init_llm_client(llm_provider)

    def generate_from_production(self, k: int = 50) -> List[SynthesizedLesson]:
        """
        Generate lessons from high-priority production bugs.

        Args:
            k: Number of bugs to sample

        Returns:
            List of synthesized lessons
        """
        print(f"[AutoCurriculum] Generating lessons from top {k} priority bugs")

        # Sample high-priority experiences from ReplayStore (REUSES existing)
        experiences = self.replay_store.sample(k=k, strategy="priority")

        # Filter to production bugs only
        production_bugs = [
            exp for exp in experiences
            if exp.metadata.get('provenance') == 'production_bug'
        ]

        print(f"[AutoCurriculum] Found {len(production_bugs)} production bugs")

        lessons = []
        for exp in production_bugs:
            self.metrics.bugs_processed += 1

            # Synthesize lesson from bug
            lesson = self._synthesize_lesson(exp)
            if not lesson:
                continue

            # Validate using existing LessonValidator (REUSES existing)
            if self.validator.is_duplicate(lesson.name):
                print(f"[AutoCurriculum] Skipping duplicate: {lesson.name}")
                self.metrics.lessons_rejected += 1
                continue

            lessons.append(lesson)
            self.metrics.lessons_generated += 1

            # Update metrics
            self._update_metrics(lesson)

        print(f"[AutoCurriculum] Generated {len(lessons)} lessons "
              f"({self.metrics.lessons_rejected} duplicates rejected)")

        self.metrics.last_generation_time = datetime.now().isoformat()

        return lessons

    def _synthesize_lesson(self, experience: Experience) -> Optional[SynthesizedLesson]:
        """
        Synthesize a lesson from a production bug experience.

        Args:
            experience: Production bug experience

        Returns:
            Synthesized lesson or None if synthesis fails
        """
        metadata = experience.metadata

        # Extract bug information
        source_code = metadata.get('source_code', '')
        bug_type = metadata.get('bug_type', 'unknown')
        severity = metadata.get('severity', 'medium')
        language = metadata.get('language', 'python')

        if not source_code:
            return None

        # Generate lesson name
        lesson_name = self._generate_lesson_name(bug_type, experience.task_id)

        # Synthesize fixed code (would use LLM in production)
        before_code, after_code = self._synthesize_fix(source_code, bug_type)
        if not before_code or not after_code:
            return None

        # Generate test code
        test_code = self._synthesize_test(before_code, after_code, bug_type)
        if not test_code:
            return None

        # Determine difficulty based on bug complexity
        difficulty = self._estimate_difficulty(bug_type, severity)

        # Generate description
        description = self._generate_description(bug_type, severity)

        # Create lesson
        lesson = SynthesizedLesson(
            name=lesson_name,
            description=description,
            before_code=before_code,
            after_code=after_code,
            test_code=test_code,
            language=language,
            difficulty=difficulty,
            category=bug_type,
            tags=[bug_type, severity, 'production', 'auto_generated'],
            provenance='production_bug',
            source_bug_id=experience.task_id,
            surprise_score=experience.surprise or 0.0
        )

        return lesson

    def _init_llm_client(self, provider: str):
        """Initialize LLM client based on provider"""
        import os

        if provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("NERION_V2_GEMINI_KEY")
                if not api_key:
                    print("[AutoCurriculum] Warning: NERION_V2_GEMINI_KEY not set, LLM disabled")
                    return None
                genai.configure(api_key=api_key)
                return genai.GenerativeModel('gemini-pro')
            except ImportError:
                print("[AutoCurriculum] google-generativeai not installed, LLM disabled")
                return None
        else:
            print(f"[AutoCurriculum] Unsupported LLM provider: {provider}")
            return None

    def _synthesize_fix(
        self,
        buggy_code: str,
        bug_type: str
    ) -> tuple[str, str]:
        """
        Synthesize fix for buggy code.

        Args:
            buggy_code: Code with bug
            bug_type: Type of bug

        Returns:
            (before_code, after_code)
        """
        if self.llm_client is None:
            # Fallback: return code as-is (will be rejected by quality checks)
            return buggy_code, buggy_code

        try:
            prompt = f"""Fix this {bug_type} bug in the following Python code.

Buggy code:
```python
{buggy_code}
```

Provide:
1. The ORIGINAL buggy code (unchanged)
2. The FIXED code

Format your response exactly as:
BEFORE:
```python
<original buggy code>
```

AFTER:
```python
<fixed code>
```
"""

            response = self.llm_client.generate_content(prompt)
            text = response.text

            # Parse response
            before_marker = "BEFORE:"
            after_marker = "AFTER:"

            if before_marker in text and after_marker in text:
                before_idx = text.index(before_marker) + len(before_marker)
                after_idx = text.index(after_marker)
                after_start = after_idx + len(after_marker)

                before_section = text[before_idx:after_idx].strip()
                after_section = text[after_start:].strip()

                # Extract code from markdown blocks
                before_code = self._extract_code_from_markdown(before_section) or buggy_code
                after_code = self._extract_code_from_markdown(after_section) or buggy_code

                # Validate that codes are different
                if before_code.strip() == after_code.strip():
                    print("[AutoCurriculum] LLM generated identical before/after code")
                    return buggy_code, buggy_code

                return before_code, after_code
            else:
                print("[AutoCurriculum] LLM response format error")
                return buggy_code, buggy_code

        except Exception as e:
            print(f"[AutoCurriculum] LLM fix synthesis failed: {e}")
            return buggy_code, buggy_code

    def _extract_code_from_markdown(self, text: str) -> Optional[str]:
        """Extract code from markdown code block"""
        import re
        # Match ```python ... ``` or ``` ... ```
        pattern = r'```(?:python)?\s*(.*?)\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no code block, return text as-is (after stripping)
        return text.strip() if text.strip() else None

    def _synthesize_test(
        self,
        before_code: str,
        after_code: str,
        bug_type: str
    ) -> str:
        """
        Synthesize test that catches the bug.

        Args:
            before_code: Buggy code
            after_code: Fixed code
            bug_type: Type of bug

        Returns:
            Test code
        """
        if self.llm_client is None:
            # Fallback: return minimal placeholder test
            return f"""
import pytest

def test_{bug_type.replace(' ', '_')}():
    # Test that catches {bug_type}
    result = function_under_test()
    assert result is not None
"""

        try:
            prompt = f"""Generate a pytest test that catches the bug in the following code.

BUGGY CODE (should fail test):
```python
{before_code}
```

FIXED CODE (should pass test):
```python
{after_code}
```

BUG TYPE: {bug_type}

Requirements:
1. The test should FAIL when run against the buggy code
2. The test should PASS when run against the fixed code
3. Use pytest framework
4. Include all necessary imports
5. The test function should have a descriptive name
6. Include docstring explaining what the test checks

Format your response as:
TEST:
```python
<complete test code>
```
"""

            response = self.llm_client.generate_content(prompt)
            text = response.text

            # Extract test code
            if "TEST:" in text:
                test_idx = text.index("TEST:") + len("TEST:")
                test_section = text[test_idx:].strip()
                test_code = self._extract_code_from_markdown(test_section)

                if test_code and len(test_code.strip()) > 50:
                    # Validate test has basic pytest structure
                    if "def test_" in test_code and "assert" in test_code:
                        return test_code

            # If parsing failed, try to extract any code block
            test_code = self._extract_code_from_markdown(text)
            if test_code and "def test_" in test_code and "assert" in test_code:
                return test_code

            # Fallback if LLM response was invalid
            print("[AutoCurriculum] LLM test synthesis produced invalid format")
            return self._generate_fallback_test(before_code, after_code, bug_type)

        except Exception as e:
            print(f"[AutoCurriculum] LLM test synthesis failed: {e}")
            return self._generate_fallback_test(before_code, after_code, bug_type)

    def _generate_fallback_test(
        self,
        before_code: str,
        after_code: str,
        bug_type: str
    ) -> str:
        """
        Generate a basic fallback test when LLM synthesis fails.

        Args:
            before_code: Buggy code
            after_code: Fixed code
            bug_type: Type of bug

        Returns:
            Minimal test code
        """
        # Try to extract function name from code
        import re
        func_match = re.search(r'def\s+(\w+)\s*\(', before_code)
        func_name = func_match.group(1) if func_match else "function_under_test"

        return f"""
import pytest

def test_{bug_type.replace(' ', '_').replace('-', '_')}():
    \"\"\"Test that catches {bug_type} bug.\"\"\"
    # This is a fallback test - LLM synthesis failed
    # TODO: Implement proper test case
    from code_under_test import {func_name}

    # Basic smoke test
    result = {func_name}()
    assert result is not None
"""

    def _generate_lesson_name(self, bug_type: str, task_id: str) -> str:
        """Generate unique lesson name"""
        # Create short hash from task_id
        short_hash = hashlib.sha256(task_id.encode()).hexdigest()[:8]
        return f"production_{bug_type.replace(' ', '_')}_{short_hash}"

    def _estimate_difficulty(self, bug_type: str, severity: str) -> str:
        """
        Estimate CERF difficulty level from bug type and severity.

        Args:
            bug_type: Type of bug
            severity: Severity level

        Returns:
            CERF level (A1-C2)
        """
        # Mapping rules (simplified)
        difficulty_map = {
            ('syntax_error', 'low'): 'A1',
            ('syntax_error', 'medium'): 'A2',
            ('logic_error', 'low'): 'B1',
            ('logic_error', 'medium'): 'B1',
            ('logic_error', 'high'): 'B2',
            ('security', 'medium'): 'B2',
            ('security', 'high'): 'C1',
            ('security', 'critical'): 'C2',
            ('performance', 'low'): 'B1',
            ('performance', 'high'): 'C1',
            ('concurrency', 'medium'): 'C1',
            ('concurrency', 'high'): 'C2',
        }

        return difficulty_map.get((bug_type, severity), 'B1')

    def _generate_description(self, bug_type: str, severity: str) -> str:
        """Generate lesson description"""
        return (
            f"Real-world {bug_type} bug (severity: {severity}) "
            f"discovered in production and automatically converted to training lesson."
        )

    def _update_metrics(self, lesson: SynthesizedLesson):
        """Update generation metrics"""
        # By bug type
        if lesson.category not in self.metrics.by_bug_type:
            self.metrics.by_bug_type[lesson.category] = 0
        self.metrics.by_bug_type[lesson.category] += 1

        # By difficulty
        if lesson.difficulty not in self.metrics.by_difficulty:
            self.metrics.by_difficulty[lesson.difficulty] = 0
        self.metrics.by_difficulty[lesson.difficulty] += 1

        # Average surprise
        n = self.metrics.lessons_generated
        self.metrics.avg_surprise = (
            (self.metrics.avg_surprise * (n - 1) + lesson.surprise_score) / n
        )

    def get_metrics(self) -> GenerationMetrics:
        """Get generation metrics"""
        return self.metrics

    def reset_metrics(self):
        """Reset metrics after reporting"""
        self.metrics = GenerationMetrics()


# Integration with existing SafeCurriculumDB
def store_lessons_in_curriculum(
    lessons: List[SynthesizedLesson],
    db_path: Path
) -> tuple[int, int]:
    """
    Store synthesized lessons in curriculum database.

    Uses SafeCurriculumDB's duplicate protection (name + SHA256 hash).

    Args:
        lessons: Synthesized lessons
        db_path: Path to curriculum.sqlite

    Returns:
        (num_added, num_rejected)
    """
    from nerion_digital_physicist.db.safe_curriculum import SafeCurriculumDB

    db = SafeCurriculumDB(str(db_path))
    num_added = 0
    num_rejected = 0

    for lesson in lessons:
        # Convert to dict for SafeCurriculumDB
        lesson_dict = {
            'name': lesson.name,
            'description': lesson.description,
            'before_code': lesson.before_code,
            'after_code': lesson.after_code,
            'test_code': lesson.test_code,
            'language': lesson.language,
            'difficulty': 'medium',  # SafeCurriculumDB expects this
            'focus_area': lesson.difficulty,  # CERF level
            'category': lesson.category,
            'tags': json.dumps(lesson.tags),
        }

        try:
            db.add_lesson(lesson_dict)
            num_added += 1
        except Exception as e:
            # Likely duplicate (SafeCurriculumDB rejects)
            print(f"[AutoCurriculum] Failed to add lesson {lesson.name}: {e}")
            num_rejected += 1

    print(f"[AutoCurriculum] Stored in curriculum: "
          f"{num_added} added, {num_rejected} rejected")

    return num_added, num_rejected


# Example usage for testing
def example_workflow():
    """Example workflow for auto-curriculum generation"""
    from pathlib import Path

    # Setup
    replay_root = Path("data/replay")
    curriculum_path = Path("out/learning/curriculum.sqlite")

    # Initialize components (REUSES existing infrastructure)
    replay_store = ReplayStore(replay_root)
    validator = LessonValidator()
    generator = AutoCurriculumGenerator(replay_store, validator)

    # Generate lessons from production
    lessons = generator.generate_from_production(k=50)

    # Store in curriculum
    num_added, num_rejected = store_lessons_in_curriculum(lessons, curriculum_path)

    # Print metrics
    metrics = generator.get_metrics()
    print(f"\n[AutoCurriculum] Generation complete:")
    print(f"  Bugs processed: {metrics.bugs_processed}")
    print(f"  Lessons generated: {metrics.lessons_generated}")
    print(f"  Lessons rejected: {metrics.lessons_rejected}")
    print(f"  Stored in curriculum: {num_added}")
    print(f"  Average surprise: {metrics.avg_surprise:.4f}")


if __name__ == "__main__":
    example_workflow()
