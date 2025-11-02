"""
Learning subsystem for the Digital Physicist.

This package contains the core components of the autonomous learning system:
- InspirationSelector: Chooses lesson topics based on CEFR framework
- IdeaGenerator: Generates concrete lesson ideas using LLM providers
- ImpactAssessor: Evaluates the viability and impact of lesson ideas
- LessonValidator: Validates lessons for duplicates and policy compliance
- SpecializedLessonGenerators: Specialized generators for all CEFR levels (A1-C2)
- ContrastiveLearner: Self-supervised contrastive learning for code embeddings
- CodeAugmentor: Semantic-preserving code transformations
- DistributedLearner: Knowledge sharing and distributed learning across agents
- KnowledgeBase: Shared knowledge repository for multi-agent learning
"""

from .inspiration_selector import InspirationSelector
from .idea_generator import IdeaGenerator
from .impact_assessor import ImpactAssessor
from .lesson_validator import LessonValidator
from .specialized_lesson_generators import SpecializedLessonGenerators
from .contrastive import (
    ContrastiveLearner,
    ContrastiveEncoder,
    ContrastiveTrainingConfig,
    ContrastiveLoss,
    SimpleFeatureExtractor
)
from .augmentation import (
    CodeAugmentor,
    AugmentationType,
    AugmentationResult
)
from .distributed import (
    DistributedLearner,
    KnowledgeBase,
    KnowledgeItem,
    LearningExperience
)

__all__ = [
    "InspirationSelector",
    "IdeaGenerator",
    "ImpactAssessor",
    "LessonValidator",
    "SpecializedLessonGenerators",
    "ContrastiveLearner",
    "ContrastiveEncoder",
    "ContrastiveTrainingConfig",
    "ContrastiveLoss",
    "SimpleFeatureExtractor",
    "CodeAugmentor",
    "AugmentationType",
    "AugmentationResult",
    "DistributedLearner",
    "KnowledgeBase",
    "KnowledgeItem",
    "LearningExperience",
]
