from __future__ import annotations

import json
from pathlib import Path


def _build_parser():
    import selfcoder.cli as cli
    return cli._build_parser()


def test_patch_hunks_preview_and_apply(tmp_path, capsys):
    # Prepare a target file and a plan that adds a module docstring (single hunk)
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    plan = {
        "actions": [
            {"kind": "add_module_docstring", "payload": {"doc": "X"}},
        ],
        "target_file": str(target),
    }
    planfile = tmp_path / "plan.json"
    planfile.write_text(json.dumps(plan), encoding="utf-8")

    parser = _build_parser()
    # Preview hunks should show at least HUNK 0
    ns = parser.parse_args(["patch", "preview-hunks", str(planfile), "--file", str(target)])
    rc = ns.func(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "HUNK 0" in out

    # Apply only that hunk and verify file content changed (docstring present)
    ns = parser.parse_args(["patch", "apply-hunks", str(planfile), "--file", str(target), "--hunk", "0"])
    rc = ns.func(ns)
    assert rc == 0
    text = target.read_text(encoding="utf-8")
    assert '"""X"""' in text


def test_voice_metrics_cli(tmp_path, monkeypatch, capsys):
    out_dir = Path("out/voice")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "latency.jsonl"
    rows = [
        {"ts": 0, "backend": "google", "model": "small", "duration_ms": 120},
        {"ts": 1, "backend": "sphinx", "model": "base", "duration_ms": 80},
    ]
    with log_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    parser = _build_parser()
    ns = parser.parse_args(["voice", "metrics", "--last", "10"])
    rc = ns.func(ns)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data.get("samples", 0) >= 2
    assert isinstance((data.get("groups") or []), list)


def test_health_dashboard_cli(tmp_path, capsys):
    # Fake experience log and voice latency + minimal coverage.json
    exp_dir = Path("out/experience"); exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "log.jsonl").write_text(
        json.dumps({"outcome_success": True, "action_taken": {"steps": []}}) + "\n",
        encoding="utf-8",
    )
    vdir = Path("out/voice"); vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "latency.jsonl").write_text(
        json.dumps({"backend": "google", "model": "small", "duration_ms": 50}) + "\n",
        encoding="utf-8",
    )
    Path("coverage.json").write_text(json.dumps({"totals": {"covered_lines": 1, "num_statements": 1}}), encoding="utf-8")
    parser = _build_parser()
    ns = parser.parse_args(["health", "dashboard", "--json", "--last", "10"])
    rc = ns.func(ns)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "experience" in data and "stt" in data and "coverage" in data


def test_health_html_generation(tmp_path, capsys):
    # Ensure experience/voice logs exist so HTML has content
    out_dir = Path('out/experience'); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'log.jsonl').write_text(json.dumps({"outcome_success": True, "action_taken": {"steps": []}})+"\n", encoding='utf-8')
    vdir = Path('out/voice'); vdir.mkdir(parents=True, exist_ok=True)
    (vdir / 'latency.jsonl').write_text(json.dumps({"backend":"google","model":"small","duration_ms":50})+"\n", encoding='utf-8')
    html_path = tmp_path / 'health.html'
    parser = _build_parser()
    ns = parser.parse_args(["health", "html", "--out", str(html_path), "--last", "5", "--refresh", "1"])
    rc = ns.func(ns)
    assert rc == 0
    assert html_path.exists()
    content = html_path.read_text(encoding='utf-8')
    assert 'Nerion Health Dashboard' in content


def test_artifacts_export_cli(tmp_path, capsys):
    # Create a chunk and export a markdown report
    chdir = Path("out/knowledge/chunks")
    chdir.mkdir(parents=True, exist_ok=True)
    sample = {
        "topic": "search:laptops",
        "domain": "example.com",
        "url": "https://example.com/x",
        "extract": "Battery up to 15 hours",
        "date": 0,
    }
    (chdir / "x.json").write_text(json.dumps(sample), encoding="utf-8")
    parser = _build_parser()
    ns = parser.parse_args(["artifacts", "export", "--topic", "search:laptops"])
    rc = ns.func(ns)
    assert rc == 0
    out = capsys.readouterr().out
    # Check headings and citations present
    assert "# Topic:" in out
    assert "## Citations" in out


def test_reviewer_style_threshold_blocks(tmp_path, monkeypatch):
    # Plan with a very long module docstring line to trigger style hints
    target = tmp_path / "mod2.py"
    target.write_text("x=1\n", encoding="utf-8")
    long_doc = "A" * 150
    plan = {
        "actions": [
            {"kind": "add_module_docstring", "payload": {"doc": long_doc}},
        ],
        "target_file": str(target),
    }
    # Enforce style hints limit to 0
    monkeypatch.setenv("NERION_REVIEW_STYLE_MAX", "0")
    from selfcoder.orchestrator import apply_plan
    touched = apply_plan(plan, dry_run=False)
    # Should block and return []
    assert touched == []
