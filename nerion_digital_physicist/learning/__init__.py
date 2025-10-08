"""
Learning subsystem for the Digital Physicist.

This package contains the core components of the autonomous learning system:
- InspirationSelector: Chooses lesson topics based on CEFR framework
- IdeaGenerator: Generates concrete lesson ideas using LLM providers
- ImpactAssessor: Evaluates the viability and impact of lesson ideas
- LessonValidator: Validates lessons for duplicates and policy compliance
- SpecializedLessonGenerators: Specialized generators for all CEFR levels (A1-C2)
"""

from .inspiration_selector import InspirationSelector
from .idea_generator import IdeaGenerator
from .impact_assessor import ImpactAssessor
from .lesson_validator import LessonValidator
from .specialized_lesson_generators import SpecializedLessonGenerators

__all__ = [
    "InspirationSelector",
    "IdeaGenerator",
    "ImpactAssessor",
    "LessonValidator",
    "SpecializedLessonGenerators",
]
