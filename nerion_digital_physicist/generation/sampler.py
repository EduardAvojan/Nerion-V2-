"""Template sampling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import random


@dataclass
class TemplateSpec:
    template_id: str
    weight: float = 1.0


class TemplateSampler:
    def __init__(self, templates: Sequence[TemplateSpec], seed: int | None = None):
        if not templates:
            raise ValueError("Template list must not be empty")
        self.templates: List[TemplateSpec] = list(templates)
        self.random = random.Random(seed)

    def sample(self) -> TemplateSpec:
        weights = [spec.weight for spec in self.templates]
        choices = [spec for spec in self.templates]
        return self.random.choices(choices, weights=weights, k=1)[0]

    def sequence(self, count: int) -> Iterable[TemplateSpec]:
        for _ in range(count):
            yield self.sample()
