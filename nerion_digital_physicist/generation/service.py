"""CLI harness for generating Phase 3 tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from time import perf_counter

if __package__ is None or __package__ == "":  # pragma: no cover - script invocation support
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from nerion_digital_physicist.generation.builder import TEMPLATE_FACTORIES, TaskBuilder
    from nerion_digital_physicist.generation.sampler import TemplateSampler, TemplateSpec
    from nerion_digital_physicist.infrastructure.memory import ReplayStore
    from nerion_digital_physicist.infrastructure.registry import ManifestRegistry
    from nerion_digital_physicist.infrastructure.telemetry import TelemetryLogger
else:
    from .builder import TEMPLATE_FACTORIES, TaskBuilder
    from .sampler import TemplateSampler, TemplateSpec
    from ..infrastructure.memory import ReplayStore
    from ..infrastructure.registry import ManifestRegistry
    from ..infrastructure.telemetry import TelemetryLogger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tasks for Nerion Phase 3")
    parser.add_argument("count", type=int, help="Number of tasks to generate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for template sampling")
    parser.add_argument(
        "--templates",
        type=str,
        default=None,
        help="Optional JSON mapping of template_id to weight",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable telemetry-driven curriculum sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "generated_tasks"),
        help="Directory where tasks should be generated",
    )
    return parser.parse_args()


def load_template_specs(template_weights: Dict[str, float] | None) -> list[TemplateSpec]:
    specs = []
    for template_id in TEMPLATE_FACTORIES:
        weight = template_weights.get(template_id, 1.0) if template_weights else 1.0
        specs.append(TemplateSpec(template_id=template_id, weight=weight))
    return specs


def main() -> None:
    args = parse_args()
    template_weights = json.loads(args.templates) if args.templates else None

    output_root = Path(args.output)
    registry = ManifestRegistry(output_root)
    telemetry = TelemetryLogger(output_root)
    replay_store = ReplayStore(output_root)
    specs = load_template_specs(template_weights)

    if args.curriculum and not template_weights:
        try:
            from .curriculum import compute_curriculum_weights
        except ImportError:  # pragma: no cover - script invocation support
            from nerion_digital_physicist.generation.curriculum import compute_curriculum_weights

        curriculum_weights = compute_curriculum_weights(output_root)
        if curriculum_weights:
            specs = load_template_specs(curriculum_weights)

    sampler = TemplateSampler(specs, seed=args.seed)
    builder = TaskBuilder(output_root, registry, telemetry=telemetry, replay=replay_store)

    run_start = perf_counter()
    manifests = []
    for spec in sampler.sequence(args.count):
        manifest = builder.build_task(spec.template_id, seed=args.seed)
        manifests.append(manifest)
        print(f"Generated task {manifest.task_id} for template {spec.template_id}")

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps([json.loads(m.to_json()) for m in manifests], indent=2),
        encoding="utf-8",
    )

    telemetry.log(
        "generation_run_complete",
        {
            "count": len(manifests),
            "templates": {
                template_id: sum(1 for m in manifests if m.template_id == template_id)
                for template_id in {m.template_id for m in manifests}
            },
            "duration_seconds": perf_counter() - run_start,
        },
    )


if __name__ == "__main__":
    main()
