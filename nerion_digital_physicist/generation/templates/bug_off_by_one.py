"""Bug injection template for off-by-one errors."""
from __future__ import annotations

from typing import Dict, Any

from .base import Template, RenderResult


class OffByOneBugTemplate(Template):
    template_id = "bug_off_by_one"
    default_parameters = {
        "bug_type": "missing_last",
        "data_shape": "list",
        "size": 6,
    }

    def render(self, parameters: Dict[str, Any]) -> RenderResult:
        bug_type = parameters.get("bug_type", self.default_parameters["bug_type"])
        size = parameters.get("size", self.default_parameters["size"])

        src_lines = [
            "def collect_values(data: list[int]) -> list[int]:",
            "    \"\"\"Return a copy of data -- intentionally buggy variant.\"\"\"",
        ]

        if bug_type == "missing_last":
            src_lines.extend([
                "    result = []",
                "    for idx in range(len(data) - 1):",
                "        result.append(data[idx])",
                "    return result",
            ])
        elif bug_type == "missing_first":
            src_lines.extend([
                "    result = []",
                "    for idx in range(1, len(data)):",
                "        result.append(data[idx])",
                "    return result",
            ])
        elif bug_type == "range_exclusive":
            src_lines.extend([
                "    return [data[idx] for idx in range(0, len(data) - 1)]",
            ])
        else:
            raise ValueError(f"Unsupported bug_type: {bug_type}")

        tests = f"""import pytest\nfrom module import collect_values\n\n\ndef test_collect_values_expected_length():\n    data = list(range({size}))\n    result = collect_values(data)\n    assert len(result) == len(data)\n\n\ndef test_collect_values_matches_input():\n    data = list(range({size}))\n    result = collect_values(data)\n    assert result == data\n"""

        docs = (
            """# Bug Fix Task\n\nThe function `collect_values` intentionally contains an off-by-one error.\nUpdate the implementation so that it returns an exact copy of the incoming list.\nAll provided tests currently fail and should pass once the defect is fixed.\n"""
        )

        return RenderResult(source_code="\n".join(src_lines) + "\n", tests=tests, docs=docs)
