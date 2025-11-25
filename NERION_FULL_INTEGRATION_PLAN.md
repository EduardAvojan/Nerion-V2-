# Nerion Full Integration Plan: From Components to Autonomous Immune System

**Created:** November 24, 2025
**Goal:** Connect all dormant components to create a self-improving software immune system
**Timeline:** 4-6 weeks for full integration
**Current State:** Gym training running (30 tasks, 81 episodes), GNN at 61.85% accuracy

---

## Executive Summary

Nerion has sophisticated components that are **built but not connected**. This plan wires them together to achieve:

1. **Contrastive Pretraining** → Better embeddings from unlabeled code
2. **Causal-Aware Graphs** → GNN learns cause-effect, not just structure
3. **MAML Adaptation** → Few-shot learning for new bug patterns
4. **Surprise-Weighted Replay** → Focus learning on hard cases
5. **Chain-of-Thought Reasoning** → Explainable, trustworthy decisions

**Target:** 90%+ accuracy with explainable reasoning and continuous improvement

---

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [Phase 1: Contrastive Pretraining](#phase-1-contrastive-pretraining)
3. [Phase 2: Causal-Aware Graphs](#phase-2-causal-aware-graphs)
4. [Phase 3: MAML Integration](#phase-3-maml-integration)
5. [Phase 4: Surprise-Weighted Replay](#phase-4-surprise-weighted-replay)
6. [Phase 5: Chain-of-Thought Reasoning](#phase-5-chain-of-thought-reasoning)
7. [Phase 6: Full System Integration](#phase-6-full-system-integration)
8. [Training Runs](#training-runs)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Rollback Plan](#rollback-plan)

---

## Current Architecture

### What's Running Now

```
daemon/nerion_daemon.py --gym
        │
        ▼ (every 10-30 seconds)
Pick random file from training_ground/
        │
        ▼
universal_fixer.py --mode evolve_*
        │
        ▼
Claude API generates improvement
        │
        ▼ (if successful)
create_graph_data_object() → AST → PyG graph
        │
        ▼
OnlineLearner.incremental_update() → EWC training
        │
        ▼
EpisodicMemory stores experience
```

### Components Used vs Dormant

| Component | Status | Location |
|-----------|--------|----------|
| OnlineLearner (EWC) | ✅ USED | `training/online_learner.py` |
| EpisodicMemory | ✅ USED | `memory/episodic_memory.py` |
| create_graph_data_object | ✅ USED | `agent/data.py` |
| CodeGraphGCN | ✅ USED | `agent/brain.py` |
| **MAML** | ❌ DORMANT | `training/maml.py` |
| **CausalAnalyzer** | ❌ DORMANT | `agent/causal_analyzer.py` |
| **ContrastiveLearning** | ❌ DORMANT | `learning/contrastive.py` |
| **CodeAugmentor** | ❌ DORMANT | `learning/augmentation.py` |
| **ProductionCollector** | ❌ DORMANT | `infrastructure/production_collector.py` |
| **KnowledgeGraph** | ❌ DORMANT | `infrastructure/knowledge_graph.py` |
| **ChainOfThought** | ❌ DORMANT | `selfcoder/reasoning/chain_of_thought.py` |
| **ExplainablePlanner** | ❌ DORMANT | `selfcoder/planner/explainable_planner.py` |
| **PatternDetector** | ❌ DORMANT | `architecture/pattern_detector.py` |
| **ArchitectureGraph** | ❌ DORMANT | `architecture/graph_builder.py` |

---

## Phase 1: Contrastive Pretraining

**Goal:** Learn powerful code embeddings from unlabeled training_ground code
**Timeline:** Week 1
**Impact:** Better base embeddings → better downstream accuracy

### Why This First

1. Code already exists (`contrastive.py`, `augmentation.py`)
2. Massive unlabeled data available (training_ground: rich, flask, httpx, etc.)
3. Doesn't disrupt current training loop
4. Foundation for all other improvements

### Implementation Steps

#### 1.1 Create Pretraining Script

**File:** `nerion_digital_physicist/training/pretrain_contrastive.py`

```python
"""
Contrastive Pretraining for Code Embeddings

Runs BEFORE supervised training. Learns embeddings from unlabeled code.
"""
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

from nerion_digital_physicist.learning.contrastive import (
    ContrastiveTrainingConfig,
    ContrastiveLoss,
    ContrastiveTrainer
)
from nerion_digital_physicist.learning.augmentation import CodeAugmentor
from nerion_digital_physicist.agent.data import create_graph_data_object
from nerion_digital_physicist.agent.brain import CodeGraphGCN


def collect_unlabeled_code(training_ground: Path) -> list[str]:
    """Collect all Python files from training ground."""
    code_files = []
    for py_file in training_ground.rglob("*.py"):
        try:
            code = py_file.read_text(encoding='utf-8')
            if len(code) > 100:  # Skip tiny files
                code_files.append(code)
        except:
            continue
    return code_files


def create_contrastive_pairs(code_files: list[str], augmentor: CodeAugmentor):
    """Create positive pairs via augmentation."""
    pairs = []
    for code in code_files:
        try:
            aug1 = augmentor.augment(code, num_augmentations=2)
            aug2 = augmentor.augment(code, num_augmentations=2)
            if aug1.ast_valid and aug2.ast_valid:
                pairs.append((aug1.augmented_code, aug2.augmented_code))
        except:
            continue
    return pairs


def pretrain_contrastive(
    training_ground: Path,
    output_path: Path,
    config: ContrastiveTrainingConfig = None
):
    """
    Run contrastive pretraining.

    Args:
        training_ground: Path to training_ground directory
        output_path: Where to save pretrained model
        config: Training configuration
    """
    config = config or ContrastiveTrainingConfig(
        embedding_dim=256,
        hidden_dim=512,
        projection_dim=128,
        batch_size=64,
        epochs=50,
        temperature=0.07
    )

    print(f"[Contrastive] Collecting unlabeled code from {training_ground}")
    code_files = collect_unlabeled_code(training_ground)
    print(f"[Contrastive] Found {len(code_files)} code files")

    print("[Contrastive] Creating augmented pairs...")
    augmentor = CodeAugmentor(seed=42)
    pairs = create_contrastive_pairs(code_files, augmentor)
    print(f"[Contrastive] Created {len(pairs)} contrastive pairs")

    # Initialize model
    model = CodeGraphGCN(
        num_node_features=800,  # 768 GraphCodeBERT + 32 AST
        hidden_channels=config.hidden_dim,
        num_classes=config.embedding_dim
    )

    # Training loop
    trainer = ContrastiveTrainer(model, config)
    trainer.train(pairs)

    # Save pretrained weights
    torch.save(model.state_dict(), output_path)
    print(f"[Contrastive] Saved pretrained model to {output_path}")

    return model


if __name__ == "__main__":
    pretrain_contrastive(
        training_ground=Path("training_ground"),
        output_path=Path("models/contrastive_pretrained.pt")
    )
```

#### 1.2 Modify OnlineLearner to Load Pretrained Weights

**File:** `nerion_digital_physicist/training/online_learner.py`

Add to `__init__`:

```python
def __init__(
    self,
    config: Optional[EWCConfig] = None,
    device: Optional[torch.device] = None,
    pretrained_path: Optional[Path] = None  # NEW
):
    # ... existing code ...

    # Load pretrained weights if available
    if pretrained_path and pretrained_path.exists():
        print(f"[OnlineLearner] Loading pretrained weights from {pretrained_path}")
        self.pretrained_weights = torch.load(pretrained_path, map_location=self.device)
    else:
        self.pretrained_weights = None
```

#### 1.3 Update universal_fixer.py

**File:** `nerion_digital_physicist/universal_fixer.py`

Modify learner initialization:

```python
# In __init__
if enable_learning:
    # Check for pretrained weights
    pretrained_path = project_root / "models" / "contrastive_pretrained.pt"

    self.learner = OnlineLearner(
        pretrained_path=pretrained_path if pretrained_path.exists() else None
    )
```

### Training Run: Contrastive Pretraining

```bash
# Step 1: Run contrastive pretraining (1-2 hours)
cd /Users/ed/Nerion-V2
python -m nerion_digital_physicist.training.pretrain_contrastive

# Expected output:
# [Contrastive] Collecting unlabeled code from training_ground
# [Contrastive] Found 2847 code files
# [Contrastive] Creating augmented pairs...
# [Contrastive] Created 2500 contrastive pairs
# [Contrastive] Epoch 1/50: Loss=2.34
# ...
# [Contrastive] Epoch 50/50: Loss=0.45
# [Contrastive] Saved pretrained model to models/contrastive_pretrained.pt

# Step 2: Restart gym with pretrained weights
# (Gym will auto-load pretrained weights)
python daemon/nerion_daemon.py --gym
```

### Success Metrics

- [ ] 2000+ contrastive pairs created
- [ ] Contrastive loss < 0.5 after 50 epochs
- [ ] Downstream accuracy improves 5-10% over random init

---

## Phase 2: Causal-Aware Graphs

**Goal:** Inject causal edges into code graphs so GNN learns cause-effect
**Timeline:** Week 2
**Impact:** GNN understands WHY code is buggy, not just THAT it's buggy

### Why This Matters

Current graph has:
- Sequence edges (statement order)
- Call edges (function calls)
- Data flow edges (variable definitions)

Missing:
- **Causal edges** (X causes Y)
- **Control dependency** (if X then Y)
- **Impact edges** (changing X affects Y)

### Implementation Steps

#### 2.1 Create Causal Graph Injector

**File:** `nerion_digital_physicist/agent/causal_graph_injector.py`

```python
"""
Injects causal edges into PyG graphs.
"""
from torch_geometric.data import Data
from nerion_digital_physicist.agent.causal_analyzer import CausalAnalyzer


def inject_causal_edges(graph: Data, source_code: str) -> Data:
    """
    Augment graph with causal edges from CausalAnalyzer.

    Args:
        graph: Original PyG graph (AST-based)
        source_code: Source code for causal analysis

    Returns:
        Graph with additional causal edges
    """
    # Run causal analysis
    analyzer = CausalAnalyzer()
    result = analyzer.analyze_code(source_code)

    if result.graph is None or len(result.graph.edges) == 0:
        return graph  # No causal edges found

    # Convert causal edges to PyG format
    causal_edges = []
    for edge in result.graph.edges:
        # Map node names to indices (requires node mapping)
        src_idx = _get_node_index(graph, edge.source)
        tgt_idx = _get_node_index(graph, edge.target)
        if src_idx is not None and tgt_idx is not None:
            causal_edges.append([src_idx, tgt_idx])

    if causal_edges:
        import torch
        causal_edge_index = torch.tensor(causal_edges, dtype=torch.long).t()

        # Concatenate with existing edges
        graph.edge_index = torch.cat([graph.edge_index, causal_edge_index], dim=1)

        # Add edge type attribute (for heterogeneous GNN)
        num_original = graph.edge_index.shape[1] - len(causal_edges)
        edge_types = torch.cat([
            torch.zeros(num_original),  # Original edges
            torch.ones(len(causal_edges))  # Causal edges
        ])
        graph.edge_type = edge_types

    return graph


def _get_node_index(graph: Data, node_name: str) -> int | None:
    """Map causal node name to graph node index."""
    # Implementation depends on how nodes are stored
    # This is a placeholder
    if hasattr(graph, 'node_names'):
        try:
            return graph.node_names.index(node_name)
        except ValueError:
            return None
    return None
```

#### 2.2 Modify data.py to Include Causal Edges

**File:** `nerion_digital_physicist/agent/data.py`

Add to `create_graph_data_object`:

```python
def create_graph_data_object(
    file_path: Path,
    embedder=None,
    include_causal: bool = True  # NEW
) -> Data:
    """Create PyG graph from source file."""

    # ... existing AST → graph code ...

    # NEW: Add causal edges
    if include_causal:
        try:
            from nerion_digital_physicist.agent.causal_graph_injector import inject_causal_edges
            source_code = file_path.read_text(encoding='utf-8')
            graph = inject_causal_edges(graph, source_code)
        except Exception as e:
            print(f"[Warning] Could not add causal edges: {e}")

    return graph
```

### Training Run: Causal-Aware Graphs

```bash
# No separate training run needed
# Causal edges are added automatically during gym training

# Verify causal edges are being added:
python -c "
from pathlib import Path
from nerion_digital_physicist.agent.data import create_graph_data_object

graph = create_graph_data_object(Path('training_ground/rich/rich/console.py'))
print(f'Nodes: {graph.num_nodes}')
print(f'Edges: {graph.edge_index.shape[1]}')
if hasattr(graph, 'edge_type'):
    causal = (graph.edge_type == 1).sum().item()
    print(f'Causal edges: {causal}')
"
```

### Success Metrics

- [ ] 20-50% of edges are causal edges
- [ ] GNN attention weights focus on causal edges
- [ ] Root cause identification improves

---

## Phase 3: MAML Integration

**Goal:** Enable few-shot learning for new bug patterns
**Timeline:** Week 3
**Impact:** Learn new bug types from 3-5 examples instead of hundreds

### Why This Matters

Production will have rare bugs. MAML enables:
- See 3-5 examples of new bug type
- Adapt GNN in ~5 gradient steps
- Generalize to similar bugs

### Implementation Steps

#### 3.1 Create MAML Training Loop

**File:** `nerion_digital_physicist/training/maml_adapter.py`

```python
"""
MAML adapter for few-shot bug pattern learning.
"""
from typing import List, Tuple
import torch
from torch_geometric.data import Data

from nerion_digital_physicist.training.maml import MAMLTrainer, MAMLConfig, MAMLTask
from nerion_digital_physicist.memory.episodic_memory import EpisodicMemory, Episode


class MAMLAdapter:
    """
    Adapts MAML for code bug learning.

    Usage:
        adapter = MAMLAdapter(base_model)

        # When new bug pattern detected
        similar_episodes = memory.recall_similar(new_bug, k=5)
        adapted_model = adapter.adapt_to_pattern(similar_episodes)
    """

    def __init__(self, base_model, config: MAMLConfig = None):
        self.config = config or MAMLConfig(
            inner_lr=0.01,
            inner_steps=5,
            support_size=3,
            query_size=2
        )
        self.trainer = MAMLTrainer(base_model, self.config)

    def adapt_to_pattern(
        self,
        episodes: List[Episode],
        pattern_name: str = "new_pattern"
    ) -> torch.nn.Module:
        """
        Adapt model to new bug pattern using few episodes.

        Args:
            episodes: Similar episodes (3-5 examples)
            pattern_name: Name for this pattern

        Returns:
            Adapted model
        """
        if len(episodes) < 3:
            print(f"[MAML] Need at least 3 episodes, got {len(episodes)}")
            return self.trainer.meta_model

        # Convert episodes to MAML task
        task = self._episodes_to_task(episodes, pattern_name)

        # Run inner loop adaptation
        adapted_model = self.trainer.adapt(task, num_steps=self.config.inner_steps)

        return adapted_model

    def _episodes_to_task(self, episodes: List[Episode], task_id: str) -> MAMLTask:
        """Convert episodes to MAML task format."""
        graphs = []
        labels = []

        for ep in episodes:
            # Create graph from before_code (buggy)
            # Label = 0 (buggy)
            # ... implementation
            pass

        # Split into support/query
        support_graphs = graphs[:self.config.support_size]
        query_graphs = graphs[self.config.support_size:]

        return MAMLTask(
            task_id=task_id,
            support_graphs=support_graphs,
            support_labels=labels[:self.config.support_size],
            query_graphs=query_graphs,
            query_labels=labels[self.config.support_size:],
            metadata={"pattern": task_id}
        )
```

#### 3.2 Integrate MAML into OnlineLearner

**File:** `nerion_digital_physicist/training/online_learner.py`

Add MAML capability:

```python
class OnlineLearner:
    def __init__(self, ...):
        # ... existing code ...

        # MAML for few-shot adaptation
        self.maml_adapter = None
        self.novel_patterns: List[str] = []

    def enable_maml(self, model):
        """Enable MAML adaptation."""
        from nerion_digital_physicist.training.maml_adapter import MAMLAdapter
        self.maml_adapter = MAMLAdapter(model)

    def detect_novel_pattern(self, graph: Data, label: int) -> bool:
        """Detect if this is a novel pattern (high surprise)."""
        if not hasattr(self, 'model') or self.model is None:
            return False

        with torch.no_grad():
            pred = self.model(graph.x, graph.edge_index, graph.batch)
            confidence = torch.sigmoid(pred).item()

        # High surprise = model very wrong
        surprise = abs(confidence - label)
        return surprise > 0.7  # Threshold for "novel"

    def adapt_if_novel(self, episodes: List, pattern_id: str):
        """Adapt model if novel pattern detected."""
        if self.maml_adapter is None:
            return

        if len(episodes) >= 3:
            print(f"[MAML] Adapting to novel pattern: {pattern_id}")
            self.model = self.maml_adapter.adapt_to_pattern(episodes, pattern_id)
            self.novel_patterns.append(pattern_id)
```

#### 3.3 Wire into universal_fixer.py

```python
# In store_learning_example()
def store_learning_example(self, ...):
    # ... existing code ...

    # Check for novel pattern
    if self.learner.detect_novel_pattern(graph, label):
        # Find similar episodes
        similar = self.memory.recall_similar(
            code_before=original_code,
            k=5
        )
        if len(similar) >= 3:
            pattern_id = f"novel_{len(self.learner.novel_patterns)}"
            self.learner.adapt_if_novel(similar, pattern_id)
```

### Training Run: MAML

```bash
# MAML adapts automatically when novel patterns detected
# Monitor in logs:

tail -f daemon.log | grep MAML

# Expected:
# [MAML] Detected novel pattern with surprise=0.85
# [MAML] Found 4 similar episodes
# [MAML] Adapting to novel pattern: novel_0
# [MAML] Inner loop step 1/5: loss=0.8
# [MAML] Inner loop step 5/5: loss=0.2
# [MAML] Adaptation complete
```

### Success Metrics

- [ ] MAML triggers on high-surprise examples
- [ ] Adaptation completes in <5 seconds
- [ ] Adapted model performs better on similar examples

---

## Phase 4: Surprise-Weighted Replay

**Goal:** Prioritize learning from hard cases (high surprise)
**Timeline:** Week 4
**Impact:** Faster improvement on cases GNN gets wrong

### Implementation Steps

#### 4.1 Create Surprise Scorer

**File:** `nerion_digital_physicist/infrastructure/surprise_scorer.py`

```python
"""
Calculate surprise scores for prioritized replay.
"""
import torch
from dataclasses import dataclass


@dataclass
class SurpriseScore:
    """Surprise metrics for an example."""
    prediction_error: float  # |predicted - actual|
    confidence: float        # Model confidence
    rarity: float           # How rare is this pattern
    combined: float         # Weighted combination

    @classmethod
    def calculate(cls, predicted: float, actual: float, rarity: float = 0.5):
        prediction_error = abs(predicted - actual)
        confidence = max(predicted, 1 - predicted)

        # High surprise = confident AND wrong
        combined = prediction_error * confidence * 2 + rarity * 0.3
        combined = min(combined, 1.0)

        return cls(
            prediction_error=prediction_error,
            confidence=confidence,
            rarity=rarity,
            combined=combined
        )


class SurpriseWeightedBuffer:
    """
    Replay buffer with surprise-weighted sampling.
    """

    def __init__(self, max_size: int = 10000, alpha: float = 0.6):
        self.max_size = max_size
        self.alpha = alpha  # Prioritization exponent
        self.buffer = []
        self.priorities = []

    def add(self, example, surprise: SurpriseScore):
        """Add example with surprise-based priority."""
        priority = (surprise.combined + 0.01) ** self.alpha

        if len(self.buffer) >= self.max_size:
            # Remove lowest priority
            min_idx = self.priorities.index(min(self.priorities))
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)

        self.buffer.append(example)
        self.priorities.append(priority)

    def sample(self, batch_size: int):
        """Sample batch weighted by priority."""
        import numpy as np

        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probs
        )
        return [self.buffer[i] for i in indices]
```

#### 4.2 Integrate into OnlineLearner

```python
# In online_learner.py

class OnlineLearner:
    def __init__(self, ...):
        # ... existing code ...

        from nerion_digital_physicist.infrastructure.surprise_scorer import (
            SurpriseWeightedBuffer,
            SurpriseScore
        )
        self.surprise_buffer = SurpriseWeightedBuffer(max_size=10000)

    def incremental_update(self, current_model, new_data, ...):
        """Update with surprise-weighted replay."""

        # Score new examples
        for graph, label in new_data:
            with torch.no_grad():
                pred = current_model(graph.x, graph.edge_index, graph.batch)
                pred_val = torch.sigmoid(pred).item()

            surprise = SurpriseScore.calculate(pred_val, label)
            self.surprise_buffer.add((graph, label), surprise)

        # Sample replay batch weighted by surprise
        replay_batch = self.surprise_buffer.sample(
            batch_size=int(self.config.batch_size * self.config.replay_ratio)
        )

        # Combine new + replay
        training_batch = new_data + replay_batch

        # ... rest of training ...
```

### Success Metrics

- [ ] High-surprise examples sampled 3-5x more often
- [ ] Error rate on hard cases decreases faster
- [ ] Buffer maintains diverse examples

---

## Phase 5: Chain-of-Thought Reasoning

**Goal:** Make GNN decisions explainable and trustworthy
**Timeline:** Week 5
**Impact:** Users understand WHY Nerion makes decisions

### Implementation Steps

#### 5.1 Create GNN-CoT Bridge

**File:** `nerion_digital_physicist/reasoning/gnn_cot_bridge.py`

```python
"""
Bridge between GNN predictions and Chain-of-Thought reasoning.
"""
from selfcoder.reasoning.chain_of_thought import (
    ChainOfThoughtReasoner,
    ReasoningResult
)


class GNNCoTBridge:
    """
    Wraps GNN predictions with Chain-of-Thought reasoning.

    Usage:
        bridge = GNNCoTBridge(gnn_model, reasoner)
        result = bridge.predict_with_reasoning(code, context)

        print(f"Prediction: {result.decision}")
        print(f"Reasoning: {result.user_explanation}")
        print(f"Confidence: {result.overall_confidence}")
    """

    def __init__(self, gnn_model, reasoner: ChainOfThoughtReasoner = None):
        self.gnn = gnn_model
        self.reasoner = reasoner or ChainOfThoughtReasoner()

    def predict_with_reasoning(
        self,
        graph,
        source_code: str,
        context: dict = None
    ) -> ReasoningResult:
        """
        Get GNN prediction with full reasoning trace.
        """
        import torch

        # Get GNN prediction
        with torch.no_grad():
            logits = self.gnn(graph.x, graph.edge_index, graph.batch)
            confidence = torch.sigmoid(logits).item()

        prediction = "buggy" if confidence < 0.5 else "correct"
        gnn_confidence = max(confidence, 1 - confidence)

        # Build context for reasoning
        task = f"Analyze this code and explain why it appears {prediction}"
        full_context = {
            "source_code": source_code,
            "gnn_prediction": prediction,
            "gnn_confidence": gnn_confidence,
            **(context or {})
        }

        # Get Chain-of-Thought reasoning
        reasoning = self.reasoner.reason_about_modification(
            task=task,
            context=full_context,
            proposed_change=f"GNN predicts code is {prediction} with {gnn_confidence:.1%} confidence"
        )

        return reasoning

    def should_apply_fix(self, reasoning: ReasoningResult) -> tuple[bool, str]:
        """
        Decide whether to apply fix based on reasoning.

        Returns:
            (should_apply, reason)
        """
        if reasoning.overall_confidence >= 0.75:
            return True, "High confidence - applying autonomously"
        elif reasoning.overall_confidence >= 0.60:
            return False, "Medium confidence - flagged for human review"
        else:
            return False, "Low confidence - requesting guidance"
```

#### 5.2 Integrate into universal_fixer.py

```python
# In universal_fixer.py

class UniversalFixer:
    def __init__(self, ...):
        # ... existing code ...

        # Chain-of-Thought reasoning
        if enable_learning:
            from nerion_digital_physicist.reasoning.gnn_cot_bridge import GNNCoTBridge
            self.cot_bridge = GNNCoTBridge(self.model)

    def _get_model_insight(self, file_path: Path) -> str:
        """Get insight with reasoning."""
        graph = create_graph_data_object(file_path)
        source_code = file_path.read_text()

        # Get reasoning
        reasoning = self.cot_bridge.predict_with_reasoning(
            graph=graph,
            source_code=source_code,
            context={"file_path": str(file_path)}
        )

        # Build insight string
        insight = f"""
GNN Analysis:
- Prediction: {reasoning.decision}
- Confidence: {reasoning.overall_confidence:.1%}
- Risks: {', '.join(reasoning.risks_identified) or 'None identified'}

Reasoning:
{reasoning.user_explanation}
"""
        return insight
```

### Success Metrics

- [ ] 100% of predictions have reasoning traces
- [ ] Reasoning matches GNN attention weights
- [ ] Users report understanding WHY decisions are made

---

## Phase 6: Full System Integration

**Goal:** Wire everything together into cohesive system
**Timeline:** Week 6
**Impact:** All components work in harmony

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FULLY INTEGRATED NERION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PERCEPTION LAYER                                                    │
│  ├── CausalAnalyzer         → Extract cause-effect                  │
│  ├── create_graph_data      → AST + causal edges                    │
│  └── Contrastive Embeddings → Pretrained understanding              │
│                                                                      │
│  REASONING LAYER                                                     │
│  ├── GNN Prediction         → Fast pattern matching                 │
│  ├── ChainOfThought         → Explainable reasoning                 │
│  └── GNNCoTBridge           → Combined prediction + reasoning       │
│                                                                      │
│  LEARNING LAYER                                                      │
│  ├── OnlineLearner (EWC)    → Incremental updates                   │
│  ├── MAMLAdapter            → Few-shot adaptation                   │
│  ├── SurpriseBuffer         → Prioritized replay                    │
│  └── EpisodicMemory         → Experience storage                    │
│                                                                      │
│  ACTION LAYER                                                        │
│  ├── ExplainablePlanner     → Plan with reasoning                   │
│  ├── Selfcoder              → Safe code modification                │
│  └── PolicyGates            → Safety checks                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Final universal_fixer.py Structure

```python
class UniversalFixer:
    """
    Fully integrated autonomous code fixer.
    """

    def __init__(self, enable_learning=True, ...):
        # Perception
        self.embedder = get_global_embedder()
        self.causal_analyzer = CausalAnalyzer()

        # Reasoning
        self.model = self._load_model()  # Contrastive pretrained
        self.cot_bridge = GNNCoTBridge(self.model)

        # Learning
        self.learner = OnlineLearner(pretrained_path=...)
        self.learner.enable_maml(self.model)
        self.memory = EpisodicMemory(...)

        # Action
        self.planner = ExplainablePlanner()

    def analyze_and_fix(self, file_path: Path) -> FixResult:
        """Full analysis and fix pipeline."""

        # 1. PERCEIVE
        source_code = file_path.read_text()
        graph = create_graph_data_object(
            file_path,
            include_causal=True  # Causal edges
        )

        # 2. REASON
        reasoning = self.cot_bridge.predict_with_reasoning(
            graph=graph,
            source_code=source_code
        )

        # 3. DECIDE
        should_fix, reason = self.cot_bridge.should_apply_fix(reasoning)

        if not should_fix:
            return FixResult(
                applied=False,
                reason=reason,
                reasoning=reasoning
            )

        # 4. PLAN
        plan = self.planner.create_plan(
            task=f"Fix: {reasoning.decision}",
            context={"source": source_code, "reasoning": reasoning}
        )

        # 5. ACT
        if plan.estimated_risk != "high":
            fixed_code = self._apply_fix(plan)

            # 6. LEARN
            self._store_and_learn(
                original=source_code,
                fixed=fixed_code,
                graph=graph,
                reasoning=reasoning
            )

            return FixResult(
                applied=True,
                fixed_code=fixed_code,
                reasoning=reasoning
            )

        return FixResult(
            applied=False,
            reason="High risk - requires human review",
            plan=plan
        )
```

---

## Training Runs

### Complete Training Pipeline

```bash
# ============================================================
# NERION FULL TRAINING PIPELINE
# ============================================================

cd /Users/ed/Nerion-V2

# ------------------------------------------------------------
# PHASE 1: Contrastive Pretraining (2-3 hours)
# ------------------------------------------------------------
echo "Phase 1: Contrastive Pretraining"
python -m nerion_digital_physicist.training.pretrain_contrastive \
    --training-ground training_ground \
    --output models/contrastive_pretrained.pt \
    --epochs 50 \
    --batch-size 64

# Verify pretraining
python -c "
import torch
m = torch.load('models/contrastive_pretrained.pt')
print(f'Pretrained model loaded: {len(m)} parameters')
"

# ------------------------------------------------------------
# PHASE 2-5: Integrated Gym Training (ongoing)
# ------------------------------------------------------------
echo "Starting integrated gym training"
python daemon/nerion_daemon.py --gym \
    --enable-causal \
    --enable-maml \
    --enable-surprise-replay \
    --enable-cot

# Monitor progress
python scripts/gym_monitor.py

# ------------------------------------------------------------
# EVALUATION (run periodically)
# ------------------------------------------------------------
echo "Running evaluation"
python -m nerion_digital_physicist.evaluation.evaluate \
    --model models/nerion_immune_brain.pt \
    --test-set evaluation/test_set.json \
    --output evaluation/results.json

# Expected output:
# Accuracy: XX.X%
# Precision: XX.X%
# Recall: XX.X%
# F1: XX.X%
# Reasoning quality: XX.X%
```

### Training Schedule

| Week | Phase | Expected Outcome |
|------|-------|-----------------|
| 1 | Contrastive Pretraining | Better base embeddings |
| 2 | Causal Edges | GNN learns cause-effect |
| 3 | MAML Integration | Few-shot adaptation working |
| 4 | Surprise Replay | Faster improvement on hard cases |
| 5 | Chain-of-Thought | Explainable predictions |
| 6 | Full Integration | All components working together |

---

## Evaluation Metrics

### Core Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Validation Accuracy | 61.85% | 90%+ | `evaluate.py` |
| Few-shot Adaptation | N/A | 80%+ with 5 examples | MAML test |
| Root Cause Accuracy | N/A | 85%+ | Causal evaluation |
| Reasoning Quality | N/A | 80%+ user comprehension | Survey |
| Autonomous Fix Rate | 0% | 75%+ | Production monitoring |

### Monitoring Dashboard

```bash
# Create monitoring script
cat > scripts/full_monitor.py << 'EOF'
"""Full system monitoring dashboard."""
import json
from pathlib import Path
from datetime import datetime

def get_metrics():
    # GNN metrics
    learner_state = torch.load('models/online_learner_state.pt')

    # Memory metrics
    memory_path = Path('data/episodic_memory')
    episodes = len(list(memory_path.glob('*.json')))

    # MAML metrics
    novel_patterns = learner_state.get('novel_patterns', [])

    # Surprise buffer
    buffer_size = learner_state.get('buffer_size', 0)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    NERION FULL SYSTEM STATUS                  ║
╠══════════════════════════════════════════════════════════════╣
║ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                         ║
╠══════════════════════════════════════════════════════════════╣
║ GNN Model                                                     ║
║   Tasks learned: {learner_state.get('task_count', 0):,}                              ║
║   Best accuracy: {learner_state.get('best_accuracy', 0):.1%}                            ║
╠══════════════════════════════════════════════════════════════╣
║ Memory                                                        ║
║   Episodes stored: {episodes:,}                                ║
║   Surprise buffer: {buffer_size:,}                             ║
╠══════════════════════════════════════════════════════════════╣
║ MAML                                                          ║
║   Novel patterns: {len(novel_patterns)}                        ║
╠══════════════════════════════════════════════════════════════╣
║ Contrastive Pretraining                                       ║
║   Status: {'✓ Loaded' if Path('models/contrastive_pretrained.pt').exists() else '✗ Not found'} ║
╚══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    import torch
    get_metrics()
EOF
```

---

## Rollback Plan

### If Things Go Wrong

```bash
# Rollback to pre-integration state

# 1. Stop gym
pkill -f "nerion_daemon.py"

# 2. Restore original model
cp models/archive/original_brain.pt models/nerion_immune_brain.pt

# 3. Disable new features
export NERION_DISABLE_CAUSAL=1
export NERION_DISABLE_MAML=1
export NERION_DISABLE_COT=1

# 4. Restart with original config
python daemon/nerion_daemon.py --gym
```

### Feature Flags

```python
# In universal_fixer.py
ENABLE_CAUSAL = os.getenv('NERION_DISABLE_CAUSAL') != '1'
ENABLE_MAML = os.getenv('NERION_DISABLE_MAML') != '1'
ENABLE_COT = os.getenv('NERION_DISABLE_COT') != '1'
ENABLE_SURPRISE = os.getenv('NERION_DISABLE_SURPRISE') != '1'
```

---

## Success Criteria

### Phase Complete Checklist

- [ ] **Phase 1:** Contrastive pretraining completes, model saved
- [ ] **Phase 2:** Causal edges appear in graphs (verify with debug output)
- [ ] **Phase 3:** MAML triggers on high-surprise examples
- [ ] **Phase 4:** Surprise buffer maintains diverse examples
- [ ] **Phase 5:** All predictions have reasoning traces
- [ ] **Phase 6:** Full pipeline runs without errors for 24 hours

### Ultimate Success

- [ ] GNN accuracy > 90%
- [ ] Autonomous fix rate > 75%
- [ ] User comprehension of reasoning > 80%
- [ ] System runs 30+ days without intervention
- [ ] Zero production incidents from bad fixes

---

## Next Steps

After reading this document:

1. **Backup current state**
   ```bash
   cp -r models/ models_backup_$(date +%Y%m%d)/
   ```

2. **Start Phase 1**
   ```bash
   python -m nerion_digital_physicist.training.pretrain_contrastive
   ```

3. **Monitor progress**
   ```bash
   python scripts/full_monitor.py
   ```

---

*Document created: November 24, 2025*
*Author: Claude (Opus 4)*
*For: Nerion Software Immune System Project*
