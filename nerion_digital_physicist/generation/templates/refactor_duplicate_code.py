"""Refactor duplicate code template."""
from __future__ import annotations

from typing import Dict, Any

from .base import Template, RenderResult


class RefactorDuplicateCodeTemplate(Template):
    template_id = "refactor_duplicate_code"
    default_parameters = {
        "duplication_count": 3,
        "use_classes": False,
        "io_shape": "dict",
    }

    def render(self, parameters: Dict[str, Any]) -> RenderResult:
        duplication_count = parameters.get(
            "duplication_count", self.default_parameters["duplication_count"]
        )
        use_classes = parameters.get("use_classes", self.default_parameters["use_classes"])
        io_shape = parameters.get("io_shape", self.default_parameters["io_shape"])

        if io_shape not in {"dict", "tuple", "dataclass"}:
            raise ValueError("Unsupported io_shape")

        payload_repr = "{\"value\": base}" if io_shape == "dict" else "(base, base * 2)"

        src_lines = [
            "def base_computation(x: int) -> int:",
            "    return x * x + 1",
            "",
        ]

        for idx in range(duplication_count):
            if use_classes:
                src_lines.extend(
                    [
                        f"class Processor{idx}:",
                        "    def run(self, value: int):",
                        "        base = base_computation(value)",
                        f"        return {payload_repr}",
                        "",
                    ]
                )
            else:
                src_lines.extend(
                    [
                        f"def process_{idx}(value: int):",
                        "    base = base_computation(value)",
                        f"    return {payload_repr}",
                        "",
                    ]
                )

        selector_body = []
        if use_classes:
            selector_body.append("    options = [Processor{i}() for i in range(%d)]" % duplication_count)
            selector_body.append(
                "    return options[index % len(options)].run(value)"
            )
        else:
            selector_body.append("    options = [%s]" % ", ".join(f"process_{i}" for i in range(duplication_count)))
            selector_body.append("    func = options[index % len(options)]")
            selector_body.append("    return func(value)")

        src_lines.extend(
            [
                "def select_processor(index: int, value: int):",
                *selector_body,
            ]
        )

        tests = f"""import pytest\nfrom module import base_computation, select_processor\n\n\ndef test_base_computation():\n    assert base_computation(3) == 10\n\n\ndef test_select_processors_equivalent():\n    outputs = []\n    for idx in range({duplication_count} * 2):\n        outputs.append(select_processor(idx, 4))\n    assert len(set(outputs)) == 1\n\ndef test_payload_shape():\n    payload = select_processor(0, 2)\n    assert payload\n"""

        docs = (
            """# Refactor Duplicate Code Task\n\nSeveral nearly-identical processors share the same logic. Consolidate the duplication into reusable helpers while preserving behavior and interface.\n"""
        )

        return RenderResult(source_code="\n".join(src_lines) + "\n", tests=tests, docs=docs)
