from __future__ import annotations
import argparse
import json
from pathlib import Path

from selfcoder.cli_ext import docs_cli


def _run_cmd(func, ns: argparse.Namespace):
    import io, sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = func(ns)
    finally:
        sys.stdout = old
    out = buf.getvalue()
    return code, json.loads(out)


def _write_sample(path: Path):
    path.write_text("# Sample Doc\n\n- first point\n- second point\n\nBody text here.", encoding="utf-8")


def test_docs_read_returns_normalized_text(tmp_path: Path):
    f = tmp_path / "sample.md"
    _write_sample(f)

    code, data = _run_cmd(docs_cli.cmd_read, argparse.Namespace(path=str(f)))
    assert code == 0
    assert data["path"].endswith("sample.md")
    assert data["ext"] == ".md"
    assert "Sample Doc" in data["text"]
    # normalized text should not contain newlines
    assert "\n" not in data["text"]


def test_docs_summarize_extracts_bullets(tmp_path: Path):
    f = tmp_path / "sample.md"
    _write_sample(f)

    code, data = _run_cmd(docs_cli.cmd_summarize, argparse.Namespace(path=str(f)))
    assert code == 0
    assert isinstance(data.get("bullets"), list)
    assert "- first point" in data["bullets"]
    assert "- second point" in data["bullets"]


def test_docs_assimilate_persists_artifact(tmp_path: Path):
    f = tmp_path / "sample.md"
    _write_sample(f)

    code, data = _run_cmd(docs_cli.cmd_assimilate, argparse.Namespace(path=str(f)))
    assert code == 0
    assert "artifact_path" in data
    ap = Path(data["artifact_path"]) 
    assert ap.exists() and ap.is_file()


import importlib.util as _iu
import pytest


def _has_module(name: str) -> bool:
    return _iu.find_spec(name) is not None


def test_docs_read_url_requires_requests_if_missing():
    if _has_module("requests"):
        pytest.skip("requests is installed; skipping missing-requests error test")
    with pytest.raises(RuntimeError):
        docs_cli.cmd_read(argparse.Namespace(path=None, url="https://example.com", timeout=1))


def test_docs_read_pdf_requires_pypdf_if_missing(tmp_path: Path):
    if _has_module("pypdf"):
        pytest.skip("pypdf is installed; skipping missing-pypdf error test")
    f = tmp_path / "dummy.pdf"
    # minimal bytes; import error occurs before parsing anyway
    f.write_bytes(b"%PDF-1.4\n%EOF\n")
    with pytest.raises(RuntimeError):
        docs_cli.cmd_read(argparse.Namespace(path=str(f)))
