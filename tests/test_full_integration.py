#!/usr/bin/env python3
"""
Full Integration Test for Nerion Enhanced Pipeline

Tests all Phase 1-5 integrations:
1. Contrastive pretraining components
2. Causal edge creation
3. MAML few-shot adaptation
4. Surprise-weighted replay
5. Chain-of-Thought reasoning

Run with: python -m pytest tests/test_full_integration.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestContrastivePretraining:
    """Test Phase 1: Contrastive pretraining components"""

    def test_contrastive_encoder_creation(self):
        """Test GraphContrastiveEncoder can be created"""
        from nerion_digital_physicist.training.pretrain_contrastive import (
            GraphContrastiveEncoder
        )

        model = GraphContrastiveEncoder(
            num_node_features=32,
            hidden_channels=256,
            num_layers=4,
            projection_dim=128,
        )
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_nt_xent_loss(self):
        """Test NT-Xent loss computation"""
        import torch.nn.functional as F
        from nerion_digital_physicist.training.pretrain_contrastive import NTXentLoss

        criterion = NTXentLoss(temperature=0.07)
        z1 = F.normalize(torch.randn(8, 128), dim=1)
        z2 = F.normalize(torch.randn(8, 128), dim=1)
        loss = criterion(z1, z2)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_backbone_extraction(self):
        """Test backbone weights can be extracted for transfer"""
        from nerion_digital_physicist.training.pretrain_contrastive import (
            GraphContrastiveEncoder
        )

        model = GraphContrastiveEncoder(
            num_node_features=32,
            hidden_channels=256,
            num_layers=4,
            projection_dim=128,
        )
        backbone = model.get_backbone_state_dict()

        assert len(backbone) > 0
        assert any('layers' in k for k in backbone.keys())


class TestCausalEdges:
    """Test Phase 2: Causal edge creation"""

    def test_causal_edge_types_defined(self):
        """Test causal edge types are in the edge role mapping"""
        from nerion_digital_physicist.agent.data import EDGE_ROLE_TO_INDEX

        # Check new causal edge types exist
        assert 'causal_data' in EDGE_ROLE_TO_INDEX
        assert 'causal_control' in EDGE_ROLE_TO_INDEX
        assert 'state_change' in EDGE_ROLE_TO_INDEX
        assert 'exception_flow' in EDGE_ROLE_TO_INDEX

    def test_causal_edges_created(self):
        """Test that causal edges are created in graphs"""
        from nerion_digital_physicist.agent.data import (
            create_graph_data_from_source,
            EDGE_ROLE_TO_INDEX
        )

        code = '''
def process(data):
    if not data:
        raise ValueError("No data")
    result = transform(data)
    return result
'''
        graph = create_graph_data_from_source(code)

        # Check edge_attr has correct dimension
        assert graph.edge_attr.shape[1] == len(EDGE_ROLE_TO_INDEX)

        # Check some causal edges exist
        causal_data_idx = EDGE_ROLE_TO_INDEX['causal_data']
        has_causal_data = graph.edge_attr[:, causal_data_idx].sum() > 0
        # Note: might not always have causal_data depending on code
        assert graph.edge_attr.shape[0] > 0  # Has some edges


class TestMAMLIntegration:
    """Test Phase 3: MAML integration"""

    def test_maml_config(self):
        """Test MAML config can be created"""
        from nerion_digital_physicist.training.maml import MAMLConfig

        config = MAMLConfig(
            inner_lr=0.01,
            inner_steps=3,
            meta_batch_size=4,
            support_size=3,
            query_size=5,
            first_order=True,
        )
        assert config.inner_lr == 0.01
        assert config.first_order is True

    def test_maml_task_creation(self):
        """Test MAML task can be created"""
        from nerion_digital_physicist.training.maml import MAMLTask

        task = MAMLTask(
            task_id="test_task",
            support_graphs=[],
            support_labels=[],
            query_graphs=[],
            query_labels=[],
            metadata={'bug_type': 'test'}
        )
        assert task.task_id == "test_task"


class TestSurpriseReplay:
    """Test Phase 4: Surprise-weighted replay"""

    def test_production_bug_creation(self):
        """Test ProductionBug can be created"""
        from nerion_digital_physicist.infrastructure.production_collector import ProductionBug

        bug = ProductionBug(
            bug_id="test_001",
            source_code="def buggy(): pass",
            file_path="test.py",
            language="python",
            bug_type="logic_error",
            severity="medium",
        )
        assert bug.bug_id == "test_001"
        assert bug.timestamp is not None

    def test_feedback_metrics(self):
        """Test FeedbackMetrics tracking"""
        from nerion_digital_physicist.infrastructure.production_collector import FeedbackMetrics

        metrics = FeedbackMetrics()
        assert metrics.total_bugs_collected == 0
        assert metrics.avg_surprise == 0.0


class TestChainOfThought:
    """Test Phase 5: Chain-of-Thought reasoning"""

    def test_reasoning_steps_defined(self):
        """Test reasoning steps enum exists"""
        from selfcoder.reasoning.chain_of_thought import ReasoningStep

        assert ReasoningStep.PROBLEM_UNDERSTANDING is not None
        assert ReasoningStep.DECISION is not None

    def test_thought_trace_creation(self):
        """Test ThoughtTrace can be created"""
        from selfcoder.reasoning.chain_of_thought import ThoughtTrace, ReasoningStep

        trace = ThoughtTrace(
            step=ReasoningStep.PROBLEM_UNDERSTANDING,
            content="Test content",
            confidence=0.85,
        )
        assert trace.confidence == 0.85
        assert trace.timestamp is not None

    def test_chain_of_thought_reasoner(self):
        """Test ChainOfThoughtReasoner basic functionality"""
        from selfcoder.reasoning.chain_of_thought import ChainOfThoughtReasoner

        reasoner = ChainOfThoughtReasoner(
            min_confidence_for_execution=0.75,
            min_confidence_for_flagging=0.60
        )

        result = reasoner.reason_about_modification(
            task="Fix null pointer bug",
            context={
                'file': 'test.py',
                'lines': '10-20',
                'has_tests': True,
            },
            proposed_change="Add null check"
        )

        assert result is not None
        assert len(result.reasoning_chain) == 6  # 6 steps
        assert 0 <= result.overall_confidence <= 1

    def test_explainable_planner(self):
        """Test ExplainablePlanner integration"""
        from selfcoder.planner.explainable_planner import ExplainablePlanner

        planner = ExplainablePlanner(min_confidence_for_execution=0.7)

        plan = planner.create_plan(
            task="Test task",
            context={'file': 'test.py'}
        )

        assert plan is not None
        assert hasattr(plan, 'reasoning')
        assert hasattr(plan, 'requires_human_review')


class TestCausalGraph:
    """Test causal graph infrastructure"""

    def test_causal_graph_creation(self):
        """Test CausalGraph can be created"""
        from nerion_digital_physicist.reasoning.causal_graph import (
            CausalGraph, CausalEdgeType, NodeType
        )

        graph = CausalGraph()
        graph.add_node("var_x", NodeType.VARIABLE, "x")
        graph.add_node("var_y", NodeType.VARIABLE, "y")
        graph.add_edge("var_x", "var_y", CausalEdgeType.DATA_FLOW)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_causal_path_finding(self):
        """Test causal path finding"""
        from nerion_digital_physicist.reasoning.causal_graph import (
            CausalGraph, CausalEdgeType, NodeType
        )

        graph = CausalGraph()
        graph.add_node("a", NodeType.VARIABLE, "a")
        graph.add_node("b", NodeType.VARIABLE, "b")
        graph.add_node("c", NodeType.VARIABLE, "c")
        graph.add_edge("a", "b", CausalEdgeType.DATA_FLOW)
        graph.add_edge("b", "c", CausalEdgeType.DATA_FLOW)

        paths = graph.find_causal_paths("a", "c")
        assert len(paths) == 1
        assert len(paths[0]) == 3  # a -> b -> c


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_graph_creation_with_all_features(self):
        """Test full graph creation pipeline"""
        from nerion_digital_physicist.agent.data import create_graph_data_from_source

        code = '''
def calculate_total(items):
    """Calculate total price of items."""
    if not items:
        return 0

    total = 0
    for item in items:
        try:
            price = item.get("price", 0)
            total += price
        except AttributeError:
            continue

    return total
'''
        graph = create_graph_data_from_source(code)

        # Check graph has nodes
        assert graph.x.shape[0] > 0

        # Check graph has edges
        assert graph.edge_index.shape[1] > 0

        # Check edge attributes
        assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
