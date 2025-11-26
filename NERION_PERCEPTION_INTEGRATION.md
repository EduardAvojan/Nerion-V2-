# Nerion Phase 0: Perception Layer Integration

**Created:** November 24, 2025
**Goal:** Integrate dormant perception components to complete the vision
**Status:** IN PROGRESS

---

## Executive Summary

The Phase 1-6 integration focused on **learning** (MAML, surprise replay, contrastive, EWC).
This Phase 0 completes the **perception** layer - giving Nerion true architectural understanding.

**Components to Integrate:**
1. `ArchitecturalGraphBuilder` - Repository-wide dependency graphs
2. `PatternDetector` - Architectural pattern recognition (MVC, Repository, etc.)
3. `KnowledgeGraph` - Persistent relationship storage
4. `CausalAnalyzer` - Root cause analysis (full integration, not just edges)

---

## Current State vs Target

### Before (Current)
```
Code File → AST → PyG Graph (nodes/edges) → GNN → Classification
                        ↓
              (causal edge types added)
```
Nerion sees: "Here's a graph with 800-dim embeddings"

### After (Target)
```
Repository → ArchitecturalGraphBuilder → Dependency Graph
                      ↓
              PatternDetector → "Flask MVC, Repository pattern"
                      ↓
Code File → CausalAnalyzer → Root causes, critical nodes, bottlenecks
                      ↓
              KnowledgeGraph → Stores relationships for RAG
                      ↓
              PyG Graph → GNN → Classification
```
Nerion sees: "Flask MVC app, auth.py depends on db.py, SQL injection risk at line 47"

---

## Integration Steps

### Step 1: Add Imports to universal_fixer.py

```python
# PERCEPTION LAYER - Phase 0 Integration
from nerion_digital_physicist.architecture.graph_builder import (
    ArchitecturalGraphBuilder, ArchitectureGraph, Module, Dependency
)
from nerion_digital_physicist.architecture.pattern_detector import (
    PatternDetector, PatternType, ArchitecturalPattern
)
from nerion_digital_physicist.infrastructure.knowledge_graph import KnowledgeGraph
from nerion_digital_physicist.agent.causal_analyzer import CausalAnalyzer, CausalAnalysisResult
```

### Step 2: Initialize Perception Components in __init__

```python
def __init__(self, ...):
    # ... existing code ...

    # PHASE 0: Perception Layer
    self._init_perception_layer()

def _init_perception_layer(self):
    """Initialize perception components for architectural understanding"""
    try:
        # Architecture graph builder
        self.arch_builder = ArchitecturalGraphBuilder()

        # Pattern detector
        self.pattern_detector = PatternDetector()

        # Knowledge graph (persistent)
        kg_path = Path(__file__).parent.parent / "data" / "knowledge_graph.graphml"
        if kg_path.exists():
            self.knowledge_graph = KnowledgeGraph.load(kg_path)
            logger.info(f"📊 Loaded knowledge graph from {kg_path}")
        else:
            self.knowledge_graph = KnowledgeGraph()
            logger.info("📊 Initialized new knowledge graph")
        self.kg_path = kg_path

        # Causal analyzer
        self.causal_analyzer = CausalAnalyzer()

        # Cache for architecture graph (expensive to rebuild)
        self._arch_graph_cache = None
        self._arch_graph_cache_time = 0

        logger.info("🔭 Perception layer initialized")
    except Exception as e:
        logger.warning(f"Could not initialize perception layer: {e}")
        self.arch_builder = None
        self.pattern_detector = None
        self.knowledge_graph = None
        self.causal_analyzer = None
```

### Step 3: Add Perception Methods

```python
def _get_architecture_understanding(self, file_path: Path) -> Dict[str, Any]:
    """
    Get deep architectural understanding of file context.

    Returns:
        Dict with patterns, dependencies, causal analysis
    """
    understanding = {
        "patterns": [],
        "dependencies": [],
        "impact_scope": [],
        "causal_analysis": None,
        "critical_nodes": [],
        "root_causes": []
    }

    if not self.arch_builder:
        return understanding

    try:
        # Get/build architecture graph (cached)
        arch_graph = self._get_or_build_arch_graph(file_path)

        if arch_graph:
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns(arch_graph)
            understanding["patterns"] = [
                {
                    "type": p.pattern_type.value,
                    "confidence": p.confidence,
                    "modules": p.modules[:5],  # Top 5
                    "violations": p.violations
                }
                for p in patterns
            ]

            # Get module for this file
            module_name = self._file_to_module_name(file_path)

            if module_name:
                # Compute impact scope
                impact = arch_graph.compute_impact(module_name)
                understanding["impact_scope"] = list(impact)[:10]

                # Find related modules
                related = arch_graph.find_related_modules(module_name)
                understanding["dependencies"] = list(related)[:10]

        # Run causal analysis on file
        source_code = file_path.read_text(encoding='utf-8')
        causal_result = self.causal_analyzer.analyze_code(source_code, str(file_path))

        understanding["causal_analysis"] = {
            "num_nodes": len(causal_result.graph.nodes),
            "num_edges": len(causal_result.graph.edges),
            "cycles": len(causal_result.cycles)
        }
        understanding["critical_nodes"] = [n.name for n in causal_result.critical_nodes[:5]]
        understanding["root_causes"] = [(n.name, d) for n, d in causal_result.root_causes[:5]]

        # Store in knowledge graph
        self._update_knowledge_graph(file_path, understanding)

    except Exception as e:
        logger.warning(f"Perception analysis failed: {e}")

    return understanding

def _get_or_build_arch_graph(self, file_path: Path) -> Optional[ArchitectureGraph]:
    """Get cached architecture graph or build new one"""
    import time

    # Cache for 5 minutes
    cache_timeout = 300

    if (self._arch_graph_cache and
        time.time() - self._arch_graph_cache_time < cache_timeout):
        return self._arch_graph_cache

    # Find repository root
    repo_root = self._find_repo_root(file_path)
    if not repo_root:
        return None

    # Build graph
    try:
        self._arch_graph_cache = self.arch_builder.build_from_directory(
            repo_root,
            max_files=500  # Limit for performance
        )
        self._arch_graph_cache_time = time.time()
        return self._arch_graph_cache
    except Exception as e:
        logger.warning(f"Could not build architecture graph: {e}")
        return None

def _find_repo_root(self, file_path: Path) -> Optional[Path]:
    """Find repository root from file path"""
    current = file_path.parent
    while current != current.parent:
        if (current / ".git").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    return file_path.parent

def _file_to_module_name(self, file_path: Path) -> Optional[str]:
    """Convert file path to Python module name"""
    try:
        # Simple conversion: path/to/file.py -> path.to.file
        parts = file_path.parts
        # Find src or package root
        for i, part in enumerate(parts):
            if part in ['src', 'lib'] or part.endswith('_ground'):
                parts = parts[i+1:]
                break

        module_parts = list(parts[:-1]) + [file_path.stem]
        return '.'.join(module_parts)
    except:
        return None

def _update_knowledge_graph(self, file_path: Path, understanding: Dict[str, Any]):
    """Update knowledge graph with new understanding"""
    if not self.knowledge_graph:
        return

    try:
        file_id = str(file_path)

        # Add file node
        self.knowledge_graph.add_node(
            file_id,
            "File",
            patterns=str(understanding.get("patterns", [])),
            critical_nodes=str(understanding.get("critical_nodes", []))
        )

        # Add dependency edges
        for dep in understanding.get("dependencies", []):
            self.knowledge_graph.add_edge(file_id, dep, "DEPENDS_ON")

        # Add impact edges
        for impacted in understanding.get("impact_scope", []):
            self.knowledge_graph.add_edge(file_id, impacted, "IMPACTS")

        # Save periodically
        if hasattr(self, '_kg_save_counter'):
            self._kg_save_counter += 1
        else:
            self._kg_save_counter = 1

        if self._kg_save_counter % 10 == 0:
            self.knowledge_graph.save(self.kg_path)

    except Exception as e:
        logger.warning(f"Knowledge graph update failed: {e}")
```

### Step 4: Integrate into Fix Pipeline

Modify `_get_model_insight()` to include perception:

```python
def _get_model_insight(self, file_path: Path) -> str:
    """Get AI insight with perception layer understanding"""

    # Existing GNN insight
    gnn_insight = self._get_gnn_insight(file_path)

    # NEW: Perception layer insight
    perception = self._get_architecture_understanding(file_path)

    insight_parts = [gnn_insight]

    # Add pattern information
    if perception["patterns"]:
        patterns_str = ", ".join([
            f"{p['type']} ({p['confidence']:.0%})"
            for p in perception["patterns"]
        ])
        insight_parts.append(f"Detected patterns: {patterns_str}")

        # Add violations
        for p in perception["patterns"]:
            if p["violations"]:
                insight_parts.append(f"Pattern violations: {', '.join(p['violations'])}")

    # Add dependency context
    if perception["dependencies"]:
        insight_parts.append(f"Related modules: {', '.join(perception['dependencies'][:5])}")

    # Add impact scope
    if perception["impact_scope"]:
        insight_parts.append(f"Changes here affect: {', '.join(perception['impact_scope'][:5])}")

    # Add causal analysis
    if perception["critical_nodes"]:
        insight_parts.append(f"Critical nodes: {', '.join(perception['critical_nodes'])}")

    if perception["root_causes"]:
        causes = [f"{name} (depth {d})" for name, d in perception["root_causes"][:3]]
        insight_parts.append(f"Root causes: {', '.join(causes)}")

    return "\n".join(insight_parts)
```

### Step 5: Use Causal Analysis for Root Cause Identification

Add to the fix generation prompt:

```python
def apply_fix(self, error: ExecutionError, fix_plan: Dict[str, Any]) -> bool:
    # ... existing code ...

    # NEW: Add causal analysis to prompt
    if self.causal_analyzer:
        try:
            source_code = file_path.read_text()
            causal_result = self.causal_analyzer.analyze_code(source_code)

            # Identify root cause of error variable
            error_var = self._extract_error_variable(error.error_msg)
            if error_var:
                root_causes = self.causal_analyzer.identify_root_cause(
                    error_var, causal_result
                )

                if root_causes:
                    prompt += "\n\nCausal Analysis:\n"
                    for node, distance, explanation in root_causes[:3]:
                        prompt += f"- Root cause: {node.name} (distance {distance})\n"
                        prompt += f"  Path: {explanation}\n"
        except Exception as e:
            logger.warning(f"Causal analysis for fix failed: {e}")

    # ... rest of fix code ...
```

---

## Testing Plan

### Test 1: Architecture Graph Building
```python
def test_arch_graph():
    builder = ArchitecturalGraphBuilder()
    graph = builder.build_from_directory(Path("training_ground/flask"))
    assert graph.get_statistics()["num_modules"] > 10
    print(f"Flask modules: {graph.get_statistics()}")
```

### Test 2: Pattern Detection
```python
def test_pattern_detection():
    builder = ArchitecturalGraphBuilder()
    graph = builder.build_from_directory(Path("training_ground/flask"))
    detector = PatternDetector()
    patterns = detector.detect_patterns(graph)
    print(detector.generate_report())
```

### Test 3: Causal Analysis
```python
def test_causal_analysis():
    analyzer = CausalAnalyzer()
    code = Path("training_ground/flask/src/flask/app.py").read_text()
    result = analyzer.analyze_code(code)
    print(f"Nodes: {len(result.graph.nodes)}")
    print(f"Root causes: {len(result.root_causes)}")
```

### Test 4: Full Integration
```python
def test_full_perception():
    fixer = UniversalFixer()
    file_path = Path("training_ground/flask/src/flask/app.py")
    understanding = fixer._get_architecture_understanding(file_path)
    assert "patterns" in understanding
    assert "causal_analysis" in understanding
```

---

## Success Metrics

- [ ] Architecture graph builds in < 30 seconds for training_ground
- [ ] Pattern detection finds MVC/Layered in Flask
- [ ] Causal analysis extracts meaningful root causes
- [ ] Knowledge graph persists between runs
- [ ] Full perception adds < 2 seconds to fix latency
- [ ] `_get_model_insight()` includes architectural context

---

## Implementation Order

1. **Add imports** to universal_fixer.py
2. **Add `_init_perception_layer()`** method
3. **Add `_get_architecture_understanding()`** method
4. **Add helper methods** (`_get_or_build_arch_graph`, etc.)
5. **Modify `_get_model_insight()`** to include perception
6. **Add causal analysis** to fix prompt
7. **Test each component** individually
8. **Test full integration**

---

*Document created: November 24, 2025*
*For: Completing Nerion's Vision - Phase 0 Perception Layer*
