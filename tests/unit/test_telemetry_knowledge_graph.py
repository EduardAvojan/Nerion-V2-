from __future__ import annotations

import json
from pathlib import Path

from ops.telemetry.knowledge_graph import (
    build_knowledge_graph,
    knowledge_hotspots,
    load_knowledge_graph,
    write_knowledge_graph,
)
from ops.telemetry.operator import summarize_snapshot


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_knowledge_graph_basic(tmp_path):
    repo = tmp_path
    (repo / "app").mkdir()
    (repo / "core").mkdir()

    _write(repo / "app" / "main.py", "import core.utils\nfrom . import helpers\n")
    _write(repo / "app" / "helpers.py", "def helper():\n    return 1\n")
    _write(repo / "core" / "utils.py", "VALUE = 1\n")

    graph = build_knowledge_graph(root=repo, include_git=False, telemetry_events_limit=10)

    assert {node.label for node in graph.nodes} == {"app", "core"}
    assert graph.stats["component_count"] == 2

    edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
    assert ("component:app", "component:core") in edge_pairs

    target = repo / "graph.json"
    path = write_knowledge_graph(graph, target)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stats"]["component_count"] == 2
    assert payload["nodes"][0]["kind"] == "component"


def test_knowledge_hotspots_and_load(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "ops").mkdir()
    _write(repo / "ops" / "telemetry.py", "from core import utils\n")
    _write(repo / "core.py", "# root module\n")

    graph = build_knowledge_graph(root=repo, include_git=False, telemetry_events_limit=10)
    path = write_knowledge_graph(graph, repo / "graph.json")

    loaded = load_knowledge_graph(path)
    assert loaded is not None
    data, loaded_path = loaded
    assert loaded_path == path

    hotspots = knowledge_hotspots(data, limit=2)
    assert hotspots
    assert hotspots[0]["component"] in {"ops", "core"}


def test_summarize_snapshot_hotspots_surface():
    snapshot = {
        "window": {"hours": 24},
        "counts_total": 10,
        "prompt_completion_ratio": {"prompts": 5, "completions": 5},
        "providers": [],
        "anomalies": [],
        "knowledge_graph": {
            "hotspots": [
                {"component": "core", "risk_score": 4.2},
                {"component": "app", "risk_score": 3.0},
            ]
        },
    }
    summary = summarize_snapshot(snapshot)
    metric_labels = [item["label"] for item in summary["metrics"]]
    assert "Hotspot" in metric_labels
    hotspot_lines = [line for line in summary["anomalies"] if line.startswith("Hotspot")]
    assert hotspot_lines
