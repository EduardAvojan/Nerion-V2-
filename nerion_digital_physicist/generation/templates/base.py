"""Template base classes and utilities for generated tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json


@dataclass
class RenderResult:
    source_code: str
    tests: str
    docs: str | None = None


class Template:
    template_id: str
    default_parameters: Dict[str, Any]

    def render(self, parameters: Dict[str, Any]) -> RenderResult:
        raise NotImplementedError

    def render_default(self) -> RenderResult:
        return self.render(self.default_parameters)

    def dump_to_directory(self, result: RenderResult, target_dir: Path) -> None:
        src_dir = target_dir / "src"
        tests_dir = target_dir / "tests"
        docs_dir = target_dir / "docs"

        src_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)
        if result.docs:
            docs_dir.mkdir(parents=True, exist_ok=True)

        (src_dir / "module.py").write_text(result.source_code, encoding="utf-8")
        (tests_dir / "test_module.py").write_text(result.tests, encoding="utf-8")
        if result.docs:
            (docs_dir / "README.md").write_text(result.docs, encoding="utf-8")

    def write_manifest_metadata(self, target_dir: Path, parameters: Dict[str, Any]) -> None:
        meta = {
            "template_id": self.template_id,
            "parameters": parameters,
        }
        (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
