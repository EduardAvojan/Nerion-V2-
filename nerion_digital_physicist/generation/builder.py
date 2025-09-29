"""Task generation harness for Phase 3."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Callable
from time import perf_counter
import shutil
import uuid

from ..infrastructure.registry import ManifestRegistry, TaskManifest
from ..infrastructure.memory import ReplayStore
from ..infrastructure.telemetry import TelemetryLogger
from .templates import (
    ArithmeticPipelineTemplate,
    OffByOneBugTemplate,
    RefactorDuplicateCodeTemplate,
    AdvancedCurriculumTemplate,
)
from .templates.base import Template


@dataclass
class TemplateFactory:
    template_id: str
    constructor: Callable[[], Template]


TEMPLATE_FACTORIES = {
    "alg_arithmetic_pipeline": TemplateFactory(
        template_id="alg_arithmetic_pipeline",
        constructor=ArithmeticPipelineTemplate,
    ),
    "bug_off_by_one": TemplateFactory(
        template_id="bug_off_by_one",
        constructor=OffByOneBugTemplate,
    ),
    "refactor_duplicate_code": TemplateFactory(
        template_id="refactor_duplicate_code",
        constructor=RefactorDuplicateCodeTemplate,
    ),
    "advanced_curriculum": TemplateFactory(
        template_id="advanced_curriculum",
        constructor=AdvancedCurriculumTemplate,
    ),
}


def compute_checksum(data: Dict[str, str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(data):
        digest.update(key.encode("utf-8"))
        digest.update(data[key].encode("utf-8"))
    return digest.hexdigest()


class TaskBuilder:
    def __init__(
        self,
        output_root: Path,
        registry: ManifestRegistry,
        telemetry: TelemetryLogger | None = None,
        replay: ReplayStore | None = None,
    ):
        self.output_root = output_root
        self.registry = registry
        self.telemetry = telemetry
        self.replay = replay

    def build_task(
        self,
        template_id: str,
        seed: int,
        parameters: Dict[str, Any] | None = None,
    ) -> TaskManifest:
        if template_id not in TEMPLATE_FACTORIES:
            raise ValueError(f"Unknown template_id: {template_id}")

        template = TEMPLATE_FACTORIES[template_id].constructor()
        params = dict(template.default_parameters)
        if parameters:
            params.update(parameters)
        params.setdefault("seed", seed)

        manifest_id = uuid.uuid4().hex
        target_dir = self.output_root / template_id / manifest_id
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        start_time = perf_counter()
        rendered = template.render(params)
        template.dump_to_directory(rendered, target_dir)
        template.write_manifest_metadata(target_dir, params)

        checksum = compute_checksum(
            {
                "source": rendered.source_code,
                "tests": rendered.tests,
                "docs": rendered.docs or "",
            }
        )

        manifest = TaskManifest.new(
            template_id=template_id,
            seed=seed,
            parameters=params,
            artifacts_path=target_dir,
            checksum=checksum,
        )
        self.registry.append(manifest)

        if self.replay:
            self.replay.append(
                task_id=manifest.task_id,
                template_id=manifest.template_id,
                status="pending",
                surprise=None,
                metadata={
                    "checksum": manifest.checksum,
                    "artifacts_path": manifest.artifacts_path,
                    "source_path": str(target_dir / "src" / "module.py"),
                },
            )

        if self.telemetry:
            elapsed = perf_counter() - start_time
            self.telemetry.log(
                "task_generated",
                {
                    "template_id": template_id,
                    "task_id": manifest.task_id,
                    "artifacts_path": str(manifest.artifacts_path),
                    "checksum": manifest.checksum,
                    "duration_seconds": elapsed,
                    "seed": seed,
                },
            )
        return manifest
