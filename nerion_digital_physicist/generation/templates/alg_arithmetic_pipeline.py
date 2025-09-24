"""Arithmetic pipeline template."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from .base import Template, RenderResult

DEFAULT_OPERATIONS = ["add", "multiply", "clamp"]


@dataclass
class OperationConfig:
    add_value: float = 3.0
    multiply_value: float = 2.0
    clamp_min: float = 0.0
    clamp_max: float = 100.0


class ArithmeticPipelineTemplate(Template):
    template_id = "alg_arithmetic_pipeline"
    default_parameters = {
        "length": 5,
        "operations": DEFAULT_OPERATIONS,
        "allow_zero_division": False,
    }

    def __init__(self, operation_config: OperationConfig | None = None) -> None:
        self.operation_config = operation_config or OperationConfig()

    def render(self, parameters: Dict[str, Any]) -> RenderResult:
        length = parameters.get("length", self.default_parameters["length"])
        operations: List[str] = parameters.get("operations", DEFAULT_OPERATIONS)
        allow_zero_division = parameters.get(
            "allow_zero_division", self.default_parameters["allow_zero_division"]
        )

        op_cfg = self.operation_config

        src_lines = [
            "from __future__ import annotations\n",
            "\n",
            "def process_numbers(values: list[float]) -> list[float]:",
            "    \"\"\"Apply a deterministic arithmetic pipeline to the input list.\"\"\"",
            "    result: list[float] = []",
            "    for item in values:",
            "        current = float(item)",
        ]

        for op in operations:
            if op == "add":
                src_lines.append(f"        current += {op_cfg.add_value}")
            elif op == "subtract":
                src_lines.append(f"        current -= {op_cfg.add_value}")
            elif op == "multiply":
                src_lines.append(f"        current *= {op_cfg.multiply_value}")
            elif op == "divide":
                if not allow_zero_division:
                    src_lines.append("        if current == 0:")
                    src_lines.append(
                        "            raise ValueError('Division by zero not allowed in pipeline')"
                    )
                src_lines.append(f"        current /= {op_cfg.multiply_value}")
            elif op == "clamp":
                src_lines.append(
                    f"        current = max({op_cfg.clamp_min}, min({op_cfg.clamp_max}, current))"
                )
            else:
                raise ValueError(f"Unsupported operation: {op}")

        src_lines.extend(
            [
                "        result.append(current)",
                "    return result",
                "",
                "def demo_input() -> list[float]:",
                f"    return [float(i) for i in range({length})]",
            ]
        )

        test_lines = [
            "import pytest",
            "from module import process_numbers, demo_input",
            "",
            "",
            "def test_pipeline_runs():",
            "    numbers = demo_input()",
            "    output = process_numbers(numbers)",
            "    assert len(output) == len(numbers)",
            "",
            "",
            "def test_pipeline_expected_values():",
            "    numbers = [0, 1, 2]",
            "    result = process_numbers(numbers)",
            "    expected = []",
            "    for item in numbers:",
            "        current = float(item)",
        ]

        for op in operations:
            if op == "add":
                test_lines.append(f"        current += {op_cfg.add_value}")
            elif op == "subtract":
                test_lines.append(f"        current -= {op_cfg.add_value}")
            elif op == "multiply":
                test_lines.append(f"        current *= {op_cfg.multiply_value}")
            elif op == "divide":
                if not allow_zero_division:
                    test_lines.append("        if current == 0:")
                    test_lines.append(
                        "            raise ValueError('Division by zero not allowed in test expectation')"
                    )
                test_lines.append(f"        current /= {op_cfg.multiply_value}")
            elif op == "clamp":
                test_lines.append(
                    f"        current = max({op_cfg.clamp_min}, min({op_cfg.clamp_max}, current))"
                )
        test_lines.extend(
            [
                "        expected.append(current)",
                "    assert result == expected",
            ]
        )

        tests = "\n".join(test_lines) + "\n"

        docs = """# Arithmetic Pipeline Task\n\nImplement the function `process_numbers` that applies a deterministic series of arithmetic operations to an input list. The default configuration chains operations in this order: {ops}.\n""".format(
            ops=", ".join(operations)
        )

        source_code = "\n".join(src_lines) + "\n"
        return RenderResult(source_code=source_code, tests=tests, docs=docs)
