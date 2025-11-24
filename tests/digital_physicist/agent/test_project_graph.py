from pathlib import Path
import sys

import pytest

from nerion_digital_physicist.agent.project_graph import ProjectParser


@pytest.fixture
def project_parser(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").touch()
    (tmp_path / "pkg" / "module.py").write_text("def func(): pass")
    (tmp_path / "main.py").write_text("from pkg import module")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("from main import func")

    parser = ProjectParser(str(tmp_path))
    parser.parse_project()
    return parser


def test_resolve_absolute_import(project_parser: ProjectParser):
    imp = {"module": "pkg.module", "name": "func", "level": 0}
    resolved = project_parser._resolve_import(imp["module"], project_parser.project_root, imp["level"])
    expected = str(Path(project_parser.project_root) / "pkg" / "module.py")
    assert resolved == expected


def test_resolve_relative_import(project_parser: ProjectParser):
    imp = {"module": "module", "name": "func", "level": 1}
    resolved = project_parser._resolve_import(imp["module"], str(Path(project_parser.project_root) / "pkg"), imp["level"])
    assert resolved is None


def test_resolve_import_from_tests(project_parser: ProjectParser):
    imp = {"module": "main", "name": "func", "level": 0}
    resolved = project_parser._resolve_import(imp["module"], str(Path(project_parser.project_root) / "tests"), imp["level"])
    expected = str(Path(project_parser.project_root) / "main.py")
    assert resolved == expected