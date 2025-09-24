"""Task generation tooling for the Nerion Digital Physicist."""

from .builder import TaskBuilder, TEMPLATE_FACTORIES
from .curriculum import compute_curriculum_weights
from .sampler import TemplateSampler, TemplateSpec
from .service import load_template_specs

__all__ = [
    "TaskBuilder",
    "TEMPLATE_FACTORIES",
    "TemplateSampler",
    "TemplateSpec",
    "compute_curriculum_weights",
    "load_template_specs",
]
