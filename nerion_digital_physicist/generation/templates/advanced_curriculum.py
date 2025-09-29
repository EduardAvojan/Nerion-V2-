"""Advanced software engineering curriculum templates."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Any, Iterable, Sequence, Set

from .base import Template, RenderResult


class AdvancedCurriculumTemplate(Template):
    """Produce lessons aimed at advanced graduate-level tracks."""

    template_id = "advanced_curriculum"
    default_parameters = {
        "track": "auto",
    }

    def render(self, parameters: Dict[str, Any]) -> RenderResult:
        track = parameters.get("track", self.default_parameters["track"]) or "auto"
        if track == "auto":
            seed = parameters.get("seed")
            if isinstance(seed, int):
                track = "coding_master" if seed % 2 else "phd_research"
            else:
                track = "phd_research"
        if track not in {"phd_research", "coding_master"}:
            raise ValueError(f"Unsupported track: {track}")
        if track == "phd_research":
            return self._render_phd_track()
        return self._render_coding_master_track()

    # ------------------------------------------------------------------
    # Track renderers
    # ------------------------------------------------------------------
    def _render_phd_track(self) -> RenderResult:
        source = """from __future__ import annotations\n\nfrom collections import defaultdict\nfrom typing import Iterable, Dict, List, Set, Tuple\n\n\nclass CausalGraphAnalyzer:\n    \"\"\"Analyze directed graphs for research-grade causal studies.\"\"\"\n\n    def __init__(self, edges: Iterable[tuple[str, str]]) -> None:\n        graph: Dict[str, Set[str]] = defaultdict(set)\n        for src, dst in edges:\n            graph[src].add(dst)\n            graph.setdefault(dst, set())\n        self._graph = graph\n        self._cache_components: list[tuple[str, ...]] | None = None\n\n    # Public API -------------------------------------------------------\n    def strongly_connected_components(self) -> list[tuple[str, ...]]:\n        \"\"\"Return strongly connected components in lexical order.\"\"\"\n        if self._cache_components is None:\n            components, _ = self._tarjan()\n            self._cache_components = components\n        return list(self._cache_components)\n\n    def is_acyclic(self) -> bool:\n        components, component_map = self._tarjan()\n        for component in components:\n            if len(component) > 1:\n                return False\n        for src, neighbours in self._graph.items():\n            if src in neighbours:\n                return False\n        return True\n\n    def feedback_edge_candidates(self) -> list[tuple[str, str]]:\n        \"\"\"Return edges that participate in cycles (useful for cuts).\"\"\"\n        components, component_map = self._tarjan()\n        cyclic_components = {comp for comp in components if len(comp) > 1}\n        edges: Set[tuple[str, str]] = set()\n        for src, neighbours in self._graph.items():\n            for dst in neighbours:\n                comp_src = component_map[src]\n                comp_dst = component_map[dst]\n                if comp_src == comp_dst:\n                    if len(comp_src) > 1 or src == dst:\n                        edges.add((src, dst))\n        return sorted(edges)\n\n    def condensation_edges(self) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:\n        \"\"\"Return edges of the condensed DAG, component-to-component.\"\"\"\n        components, component_map = self._tarjan()\n        condensed_edges: Set[tuple[tuple[str, ...], tuple[str, ...]]] = set()\n        for src, neighbours in self._graph.items():\n            for dst in neighbours:\n                src_comp = component_map[src]\n                dst_comp = component_map[dst]\n                if src_comp != dst_comp:\n                    condensed_edges.add((src_comp, dst_comp))\n        return sorted(condensed_edges)\n\n    # Internal helpers -------------------------------------------------
    def _tarjan(self) -> tuple[list[tuple[str, ...]], Dict[str, tuple[str, ...]]]:
        index = 0
        stack: list[str] = []
        indices: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        on_stack: Set[str] = set()
        ordered_components: list[tuple[str, ...]] = []
        component_map: Dict[str, tuple[str, ...]] = {}

        def strongconnect(node: str) -> None:
            nonlocal index
            indices[node] = index
            lowlink[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for neighbour in self._graph[node]:
                if neighbour not in indices:
                    strongconnect(neighbour)
                    lowlink[node] = min(lowlink[node], lowlink[neighbour])
                elif neighbour in on_stack:
                    lowlink[node] = min(lowlink[node], indices[neighbour])

            if lowlink[node] == indices[node]:
                component_set: Set[str] = set()
                while True:
                    candidate = stack.pop()
                    on_stack.remove(candidate)
                    component_set.add(candidate)
                    if candidate == node:
                        break
                ordered = tuple(sorted(component_set))
                ordered_components.append(ordered)
                for member in component_set:
                    component_map[member] = ordered

        for vertex in sorted(self._graph):
            if vertex not in indices:
                strongconnect(vertex)

        ordered_components.sort(key=lambda comp: (len(comp), comp))
        return ordered_components, component_map

"""

        tests = """import pytest
from module import CausalGraphAnalyzer


def test_detects_cycles_and_candidates():
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("C", "D"),
        ("E", "F"),
        ("F", "E"),
    ]
    analyzer = CausalGraphAnalyzer(edges)
    components = analyzer.strongly_connected_components()
    assert ("A", "B", "C") in components
    assert ("E", "F") in components
    assert analyzer.is_acyclic() is False
    assert set(analyzer.feedback_edge_candidates()) == {
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("E", "F"),
        ("F", "E"),
    }
    condensed = analyzer.condensation_edges()
    assert (("A", "B", "C"), ("D",)) in condensed
    assert (("E", "F"), ("A", "B", "C")) not in condensed


def test_acyclic_graph_short_circuit():
    analyzer = CausalGraphAnalyzer([("input", "feature"), ("feature", "model")])
    assert analyzer.is_acyclic() is True
    assert analyzer.feedback_edge_candidates() == []
    assert analyzer.condensation_edges() == [(('input',), ('feature',)), (('feature',), ('model',))]
"""

        docs = """# Research Graph Diagnostics\n\nThis lesson emulates the causality tooling used in doctoral-level platforms.\nYou are given a `CausalGraphAnalyzer` with Tarjan-style analysis helpers.\nExtend or refactor it to preserve the guarantees tested in the suite:\n\n- Strongly-connected component discovery must be deterministic.\n- Cycles should expose candidate edges that break the loop.\n- The condensed DAG must faithfully describe component-to-component flow.\n\nThe tests rely on both numeric stability (index ordering) and structural correctness.\n\n"""

        return RenderResult(source_code=source + "\n", tests=tests, docs=docs)

    def _render_coding_master_track(self) -> RenderResult:
        source = """from __future__ import annotations\n\nfrom collections import defaultdict\nfrom typing import Dict, Iterable, List, Sequence, Set\n\n\nclass DeploymentPipeline:\n    \"\"\"Plan staged deployments with dependency and capacity constraints.\"\"\"\n\n    def __init__(self, dependencies: Dict[str, Sequence[str]]) -> None:\n        canonical: Dict[str, tuple[str, ...]] = {}\n        adjacency: Dict[str, Set[str]] = defaultdict(set)\n        all_tasks: Set[str] = set()\n\n        for stage, prereqs in dependencies.items():\n            unique = tuple(dict.fromkeys(prereqs))\n            canonical[stage] = unique\n            all_tasks.add(stage)\n            for prereq in unique:\n                adjacency[prereq].add(stage)\n                all_tasks.add(prereq)\n\n        for stage in all_tasks:\n            canonical.setdefault(stage, tuple())\n            adjacency.setdefault(stage, set())\n\n        self._dependencies = canonical\n        self._adjacency = adjacency\n\n    def plan_batches(self, concurrency_limit: int) -> list[list[str]]:\n        if concurrency_limit <= 0:\n            raise ValueError("Concurrency limit must be positive")\n\n        indegree = {stage: len(prereqs) for stage, prereqs in self._dependencies.items()}\n        available = sorted(stage for stage, degree in indegree.items() if degree == 0)\n        batches: list[list[str]] = []\n        processed: Set[str] = set()\n\n        while available:\n            batch: list[str] = []\n            next_available: list[str] = []\n            while available and len(batch) < concurrency_limit:\n                stage = available.pop(0)\n                if stage in processed:\n                    continue\n                batch.append(stage)\n                processed.add(stage)\n            if not batch:\n                break\n            batches.append(batch)\n            for stage in batch:\n                for successor in sorted(self._adjacency[stage]):\n                    indegree[successor] -= 1\n                    if indegree[successor] == 0:\n                        next_available.append(successor)\n            merged = list(dict.fromkeys(available + next_available))\n            available = sorted(merged)\n\n        if any(degree > 0 for degree in indegree.values()):\n            raise ValueError("Dependency cycle detected in deployment graph")\n\n        return batches\n\n    def linear_order(self) -> list[str]:\n        batches = self.plan_batches(max(1, len(self._dependencies)))\n        result: list[str] = []\n        for batch in batches:\n            result.extend(batch)\n        return result\n\n    def critical_stages(self) -> list[str]:\n        \"\"\"Stages with the widest fan-out (most downstream dependents).\"\"\"\n        fanout = {stage: len(children) for stage, children in self._adjacency.items()}\n        if not fanout:\n            return []\n        maximum = max(fanout.values())\n        return sorted(stage for stage, value in fanout.items() if value == maximum)\n\n"""

        tests = """import pytest
from module import DeploymentPipeline


def test_batches_respect_dependencies_and_capacity():
    dependencies = {
        "data_ingest": [],
        "knowledge_graph": [],
        "feature_store": ["data_ingest"],
        "analytics": ["data_ingest", "knowledge_graph"],
        "model_training": ["feature_store", "analytics"],
        "model_evaluation": ["model_training"],
        "monitoring": ["model_evaluation"],
        "release": ["model_evaluation"],
    }
    pipeline = DeploymentPipeline(dependencies)
    batches = pipeline.plan_batches(concurrency_limit=2)
    assert batches == [
        ["data_ingest", "knowledge_graph"],
        ["analytics", "feature_store"],
        ["model_training"],
        ["model_evaluation"],
        ["monitoring", "release"],
    ]
    assert pipeline.linear_order()[0] in {"data_ingest", "knowledge_graph"}
    assert set(pipeline.critical_stages()) == {"model_evaluation"}


def test_cycle_detection_and_invalid_capacity():
    pipeline = DeploymentPipeline({"a": ["b"], "b": ["a"]})
    with pytest.raises(ValueError):
        pipeline.plan_batches(concurrency_limit=2)
    with pytest.raises(ValueError):
        pipeline.plan_batches(concurrency_limit=0)
"""

        docs = """# Specialized Deployment Sequencer\n\nThis exercise mirrors the production playbooks used in specialized engineering masters programmes.\nYou are responsible for the dependency-aware deployment pipeline that honours concurrency limits while keeping throughput high.\n\nKey expectations:\n\n- `plan_batches` must respect both ordering constraints and the concurrency budget.\n- `linear_order` should flatten the schedule into a reproducible rollout order.\n- `critical_stages` highlights stages with the widest fan-out to help incident responders.\n\nMake sure to retain deterministic ordering so the surrounding observability stack can diff successive plans.\n\n"""

        return RenderResult(source_code=source + "\n", tests=tests, docs=docs)
