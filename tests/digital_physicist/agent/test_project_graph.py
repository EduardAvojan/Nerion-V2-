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

    parser = ProjectParser(tmp_path)
    parser.discover_and_parse()
    return parser


def test_resolve_absolute_import(project_parser: ProjectParser):
    imp = {"module": "pkg.module", "name": "func", "level": 0}
    resolved = project_parser._resolve_import(imp, project_parser.project_root)
    assert resolved == "pkg/module.py"


def test_resolve_relative_import(project_parser: ProjectParser):
    imp = {"module": "module", "name": "func", "level": 1}
    resolved = project_parser._resolve_import(imp, project_parser.project_root / "pkg")
    assert resolved == "pkg/module.py"


def test_resolve_import_from_tests(project_parser: ProjectParser):
    imp = {"module": "main", "name": "func", "level": 0}
    resolved = project_parser._resolve_import(imp, project_parser.project_root / "tests")
    assert resolved == "main.py"
