# Dataset Builder Enhancements - Proposed Improvements

## Current State
- **Node features**: 32D base + optional semantic embeddings (hash 16D or CodeBERT 768D)
- **Edge features**: 6D (one-hot for edge types)
- **Graph-level**: GraphCodeBERT 768D (already implemented)

## Proposed Enhancements (Priority Order)

### 🔥 Priority 1: Per-Node Semantic Embeddings
**Impact**: HIGH - Semantic understanding at every node level

Currently nodes use hash embeddings (16D) or optional CodeBERT. We should:
- Use CodeBERT/GraphCodeBERT for ALL nodes (not just graph-level)
- Add node-level semantic embeddings as an option
- Cache per-node embeddings to avoid recomputation

**Implementation**:
- Modify `_featurize_graph()` to use CodeBERT for all nodes
- Add `--use-node-embeddings` flag to dataset_builder.py
- Node features become: 32D structural + 768D semantic = 800D total

**Expected improvement**: +5-10% accuracy (nodes understand code semantics, not just structure)

---

### 🔥 Priority 2: Structural Depth & Position Features
**Impact**: MEDIUM-HIGH - Better hierarchical understanding

Add:
- `node_depth`: How deep in hierarchy (function=0, statement=1, expression=2)
- `position_in_sequence`: Normalized position (0.0=first, 1.0=last)
- `sibling_count`: How many siblings at same level
- `ancestor_functions`: How many function ancestors

**Implementation**:
- Add to `get_node_features()` in `data.py`
- Compute during graph traversal
- Adds ~4-5 dimensions to node features

**Expected improvement**: +2-5% accuracy (better understanding of code structure)

---

### 🔥 Priority 3: Enhanced Edge Types
**Impact**: MEDIUM - More relationship types

Add new edge roles:
- `import_dependency`: Module/import relationships
- `exception_flow`: Try → except relationships
- `type_relationship`: Class inheritance, type annotations
- `scope_relationship`: Variable scope boundaries

**Implementation**:
- Extend `EDGE_ROLE_TO_INDEX` in `data.py`
- Detect during AST traversal
- Edge features become 10D (from 6D)

**Expected improvement**: +2-4% accuracy (richer relationship understanding)

---

### 🔥 Priority 4: Code Quality Metrics
**Impact**: MEDIUM - Better code quality signals

Add per-node:
- **Cognitive complexity**: More nuanced than cyclomatic (nested conditions weighted)
- **Halstead metrics**: Vocabulary, length, volume (if code snippet available)
- **Code smell indicators**: Detects common patterns (long parameter list, etc.)

**Implementation**:
- Add metrics calculator functions
- Integrate into `_collect_function_facts()` and similar
- Adds ~5-10 dimensions

**Expected improvement**: +1-3% accuracy (better quality signals)

---

### Priority 5: Graph-Level Context Features
**Impact**: LOW-MEDIUM - Global context

Add graph-level statistics as node features:
- `graph_size`: Total nodes in graph
- `avg_node_degree`: Average connectivity
- `num_functions`: Function count
- `max_depth`: Maximum nesting depth

**Implementation**:
- Compute once per graph
- Broadcast to all nodes (same value for all nodes in graph)
- Adds ~4 dimensions

**Expected improvement**: +1-2% accuracy (global context)

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add structural depth/position features (Priority 2)
2. ✅ Enhance edge types (Priority 3)
3. Test with one run

### Phase 2: Semantic Boost (3-4 hours)
4. ✅ Implement per-node CodeBERT embeddings (Priority 1)
5. ✅ Add caching for embeddings
6. ✅ Update dataset_builder.py flags

### Phase 3: Polish (2-3 hours)
7. ✅ Add code quality metrics (Priority 4)
8. ✅ Add graph-level features (Priority 5)
9. ✅ Comprehensive testing

---

## Expected Combined Impact

**Current baseline**: 64.6% (from yesterday's run)

**After all enhancements**:
- Per-node embeddings: +5-10%
- Structural features: +2-5%
- Enhanced edges: +2-4%
- Quality metrics: +1-3%
- Graph context: +1-2%

**Target**: **75-85% validation accuracy** (vs current 64.6%)

---

## Notes

- Per-node embeddings will significantly increase dataset size
- Consider making it optional (`--use-node-embeddings`)
- May need to regenerate embeddings on GPU (Colab)
- Test incrementally to measure each enhancement's impact



