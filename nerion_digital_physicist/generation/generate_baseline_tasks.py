"""Generate baseline artifacts for initial templates and register manifests."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from ..infrastructure.registry import ManifestRegistry, TaskManifest
from .templates import (
    ArithmeticPipelineTemplate,
    OffByOneBugTemplate,
    RefactorDuplicateCodeTemplate,
)

OUTPUT_ROOT = Path(__file__).resolve().parent / "generated_baselines"


def compute_checksum(*contents: str) -> str:
    digest = hashlib.sha256()
    for chunk in contents:
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def write_artifacts(template, parameters, target_dir: Path) -> str:
    result = template.render(parameters)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    template.dump_to_directory(result, target_dir)
    template.write_manifest_metadata(target_dir, parameters)
    checksum = compute_checksum(result.source_code, result.tests, result.docs or "")
    return checksum


def main() -> None:
    templates = [
        ArithmeticPipelineTemplate(),
        OffByOneBugTemplate(),
        RefactorDuplicateCodeTemplate(),
    ]

    registry = ManifestRegistry(OUTPUT_ROOT)

    for template in templates:
        params = template.default_parameters
        task_dir = OUTPUT_ROOT / template.template_id
        checksum = write_artifacts(template, params, task_dir)

        manifest = TaskManifest.new(
            template_id=template.template_id,
            seed=0,
            parameters=params,
            artifacts_path=task_dir,
            checksum=checksum,
            status="generated",
        )
        registry.append(manifest)

        manifest_path = task_dir / "manifest.json"
        manifest_path.write_text(json.dumps(json.loads(manifest.to_json()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
