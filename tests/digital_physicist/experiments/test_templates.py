from importlib import reload
from pathlib import Path

import pytest

from nerion_digital_physicist.generation.templates import (
    ArithmeticPipelineTemplate,
    OffByOneBugTemplate,
    RefactorDuplicateCodeTemplate,
)


@pytest.mark.parametrize(
    "template_cls",
    [
        ArithmeticPipelineTemplate,
        OffByOneBugTemplate,
        RefactorDuplicateCodeTemplate,
    ],
)
def test_templates_render_valid_python(template_cls) -> None:
    template = template_cls()
    result = template.render_default()

    compile(result.source_code, f"{template.template_id}/module.py", "exec")
    compile(result.tests, f"{template.template_id}/test_module.py", "exec")
    if result.docs:
        assert "#" in result.docs


def test_arithmetic_pipeline_execution() -> None:
    tpl = ArithmeticPipelineTemplate()
    rendered = tpl.render_default()
    scope = {}
    exec(rendered.source_code, scope)

    output = scope["process_numbers"]([0, 1, 2])
    assert output == [6.0, 8.0, 10.0]


def test_off_by_one_bug_present() -> None:
    tpl = OffByOneBugTemplate()
    rendered = tpl.render_default()
    scope = {}
    exec(rendered.source_code, scope)

    data = list(range(5))
    result = scope["collect_values"](data)
    assert result != data
    assert len(result) == len(data) - 1


def test_refactor_template_produces_duplicate_behaviour() -> None:
    tpl = RefactorDuplicateCodeTemplate()
    rendered = tpl.render_default()
    scope = {}
    exec(rendered.source_code, scope)

    baseline = scope["select_processor"](0, 3)
    for idx in range(1, tpl.default_parameters["duplication_count"]):
        assert scope["select_processor"](idx, 3) == baseline
