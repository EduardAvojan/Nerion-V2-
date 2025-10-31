"""
Tier 1 Integration Tests

Tests for critical integrations between Tier 1 components:
- Continuous Learner integration
- Graph Loading pipeline
- Auto-curriculum generation
- Daemon integration
- Planner integration
"""
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Skip all tests if dependencies not available
pytest_available = True
try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytest.skip("PyTorch/PyG not available", allow_module_level=True)


class TestGraphLoading:
    """Test graph loading from experiences and lessons"""

    def test_load_from_experience(self):
        """Test loading graph from Experience object"""
        from nerion_digital_physicist.utils.graph_loader import GraphLoader
        from nerion_digital_physicist.infrastructure.memory import Experience

        # Create test experience
        test_code = """
def add(a, b):
    return a + b
"""
        exp = Experience(
            experience_id="test_001",
            task_id="test_task",
            template_id="test",
            status="solved",
            metadata={'source_code': test_code}
        )

        # Load graph
        loader = GraphLoader(use_semantic_embeddings=False, cache_graphs=False)
        result = loader.load_from_experience(exp)

        assert result is not None, "Failed to load graph from experience"
        graph, label = result
        assert isinstance(graph, Data), "Graph should be PyG Data object"
        assert label == 0, "Solved experience should have label 0"
        assert graph.num_nodes > 0, "Graph should have nodes"

    def test_load_from_lesson(self):
        """Test loading graph from curriculum lesson"""
        from nerion_digital_physicist.utils.graph_loader import GraphLoader

        # Create test lesson
        lesson = {
            'name': 'test_lesson',
            'before_code': """
def buggy():
    x = 1 / 0
    return x
""",
            'after_code': """
def fixed():
    x = 1
    return x
""",
            'language': 'python'
        }

        # Load graph
        loader = GraphLoader(use_semantic_embeddings=False, cache_graphs=False)
        result = loader.load_from_lesson(lesson)

        assert result is not None, "Failed to load graph from lesson"
        graph, label = result
        assert isinstance(graph, Data), "Graph should be PyG Data object"
        assert label == 1, "Buggy code should have label 1"
        assert graph.num_nodes > 0, "Graph should have nodes"

    def test_graph_caching(self):
        """Test graph caching functionality"""
        from nerion_digital_physicist.utils.graph_loader import GraphLoader
        from nerion_digital_physicist.infrastructure.memory import Experience

        test_code = "def test(): pass"
        exp = Experience(
            experience_id="test_cache",
            task_id="test",
            template_id="test",
            status="solved",
            metadata={'source_code': test_code}
        )

        loader = GraphLoader(use_semantic_embeddings=False, cache_graphs=True)

        # Load twice
        result1 = loader.load_from_experience(exp)
        result2 = loader.load_from_experience(exp)

        assert result1 is not None and result2 is not None
        stats = loader.get_cache_stats()
        assert stats['cached_graphs'] > 0, "Cache should have entries"


class TestContinuousLearner:
    """Test continuous learner components"""

    @pytest.mark.asyncio
    async def test_model_loading(self):
        """Test model loading from registry"""
        from daemon.continuous_learner import ContinuousLearner

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_root = Path(tmpdir) / "replay"
            curriculum_path = Path(tmpdir) / "curriculum.sqlite"
            model_registry_path = Path(tmpdir) / "models"

            replay_root.mkdir()
            model_registry_path.mkdir()

            # Create empty curriculum
            import sqlite3
            conn = sqlite3.connect(str(curriculum_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    before_code TEXT,
                    after_code TEXT,
                    test_code TEXT,
                    language TEXT,
                    difficulty TEXT,
                    focus_area TEXT,
                    category TEXT,
                    tags TEXT
                )
            """)
            conn.close()

            learner = ContinuousLearner(
                replay_root=replay_root,
                curriculum_path=curriculum_path,
                model_registry_path=model_registry_path
            )

            # Test model loading (should return None when no model exists)
            model = learner._load_current_model()
            assert model is None, "Should return None when no model exists"

    def test_training_data_preparation(self):
        """Test preparing training data from experiences"""
        from daemon.continuous_learner import ContinuousLearner
        from nerion_digital_physicist.infrastructure.memory import Experience

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_root = Path(tmpdir) / "replay"
            curriculum_path = Path(tmpdir) / "curriculum.sqlite"
            model_registry_path = Path(tmpdir) / "models"

            replay_root.mkdir()
            model_registry_path.mkdir()

            # Create empty curriculum
            import sqlite3
            conn = sqlite3.connect(str(curriculum_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    before_code TEXT,
                    after_code TEXT,
                    test_code TEXT,
                    language TEXT,
                    difficulty TEXT,
                    focus_area TEXT,
                    category TEXT,
                    tags TEXT
                )
            """)
            conn.close()

            learner = ContinuousLearner(
                replay_root=replay_root,
                curriculum_path=curriculum_path,
                model_registry_path=model_registry_path
            )

            # Create test experiences
            experiences = [
                Experience(
                    experience_id=f"exp_{i}",
                    task_id=f"task_{i}",
                    template_id="test",
                    status="solved" if i % 2 == 0 else "failed",
                    metadata={'source_code': f"def test{i}(): pass"}
                )
                for i in range(5)
            ]

            # Prepare training data
            data = learner._prepare_training_data(experiences)

            # Should have loaded some graphs
            assert len(data) > 0, "Should prepare some training data"
            for graph, label in data:
                assert isinstance(graph, Data), "Should return PyG Data objects"
                assert label in [0, 1], "Labels should be 0 or 1"


class TestAutoCurriculum:
    """Test auto-curriculum generation"""

    def test_llm_client_initialization(self):
        """Test LLM client initialization"""
        from nerion_digital_physicist.curriculum.auto_generator import AutoCurriculumGenerator
        from nerion_digital_physicist.infrastructure.memory import ReplayStore
        from nerion_digital_physicist.learning import LessonValidator

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_root = Path(tmpdir) / "replay"
            replay_root.mkdir()

            replay_store = ReplayStore(replay_root)
            validator = LessonValidator()

            # Test with gemini (should gracefully handle missing API key)
            generator = AutoCurriculumGenerator(
                replay_store=replay_store,
                validator=validator,
                llm_provider="gemini"
            )

            # LLM client may be None if API key not set (expected)
            assert hasattr(generator, 'llm_client')

    def test_lesson_synthesis_without_llm(self):
        """Test lesson synthesis fallback when LLM unavailable"""
        from nerion_digital_physicist.curriculum.auto_generator import AutoCurriculumGenerator
        from nerion_digital_physicist.infrastructure.memory import ReplayStore, Experience
        from nerion_digital_physicist.learning import LessonValidator

        with tempfile.TemporaryDirectory() as tmpdir:
            replay_root = Path(tmpdir) / "replay"
            replay_root.mkdir()

            replay_store = ReplayStore(replay_root)
            validator = LessonValidator()

            generator = AutoCurriculumGenerator(
                replay_store=replay_store,
                validator=validator,
                llm_provider="none"  # No LLM
            )

            # Create test experience
            exp = Experience(
                experience_id="bug_001",
                task_id="test_bug",
                template_id="bug",
                status="failed",
                metadata={
                    'source_code': 'def bug(): return 1/0',
                    'bug_type': 'division_error',
                    'severity': 'high',
                    'provenance': 'production_bug'
                }
            )

            # Synthesize lesson
            lesson = generator._synthesize_lesson(exp)

            assert lesson is not None, "Should generate lesson even without LLM"
            assert lesson.name
            assert lesson.before_code
            assert lesson.after_code


class TestChainOfThought:
    """Test chain-of-thought reasoning"""

    def test_reasoning_pipeline(self):
        """Test complete reasoning pipeline"""
        from selfcoder.reasoning.chain_of_thought import ChainOfThoughtReasoner

        reasoner = ChainOfThoughtReasoner()

        task = "Add logging to authentication function"
        context = {
            'file': 'app/auth.py',
            'has_tests': True,
            'complexity': 'medium'
        }
        proposed_change = "Add logger.info() calls at entry and exit points"

        result = reasoner.reason_about_modification(
            task=task,
            context=context,
            proposed_change=proposed_change
        )

        assert result is not None
        assert len(result.reasoning_chain) == 6, "Should have 6 reasoning steps"
        assert result.overall_confidence > 0
        # Decision is either the proposed change or ABORT message
        assert result.decision == proposed_change or result.decision.startswith("ABORT")
        assert result.user_explanation


class TestExplainablePlanner:
    """Test explainable planner integration"""

    def test_planner_integration(self):
        """Test integration with existing planner"""
        from selfcoder.planner.explainable_planner import ExplainablePlanner

        planner = ExplainablePlanner()

        task = "Add function hello"
        context = {'file': 'test.py'}

        plan = planner.create_plan(task, context)

        assert plan is not None
        assert isinstance(plan.actions, list)
        assert plan.reasoning is not None
        assert plan.execution_strategy
        assert plan.estimated_risk in ['low', 'medium', 'high', 'critical']

    def test_plan_explanation(self):
        """Test plan explanation generation"""
        from selfcoder.planner.explainable_planner import ExplainablePlanner

        planner = ExplainablePlanner()

        task = "Fix security vulnerability"
        context = {
            'file': 'app/auth.py',
            'is_production': True,
            'complexity': 'high'
        }

        plan = planner.create_plan(task, context)

        # Get different explanation levels
        brief = planner.explain_plan(plan, detail_level='brief')
        detailed = planner.explain_plan(plan, detail_level='detailed')

        assert brief
        assert detailed
        assert len(detailed) > len(brief)


class TestCausalReasoning:
    """Test causal reasoning capabilities"""

    def test_causal_graph_analysis(self):
        """Test causal graph extraction from code"""
        from nerion_digital_physicist.agent.causal_analyzer import CausalAnalyzer

        analyzer = CausalAnalyzer()

        test_code = """
def calculate_price(quantity, discount):
    base_price = quantity * 10
    if discount:
        final_price = base_price * 0.9
    else:
        final_price = base_price
    return final_price
"""

        result = analyzer.analyze_code(test_code)

        assert result is not None
        assert result.graph is not None
        assert len(result.graph.nodes) > 0

    def test_root_cause_identification(self):
        """Test root cause identification"""
        from nerion_digital_physicist.agent.causal_analyzer import CausalAnalyzer

        analyzer = CausalAnalyzer()

        buggy_code = """
def divide(a, b):
    result = a / b
    return result
"""

        result = analyzer.analyze_code(buggy_code)
        assert result is not None

        # Identify root cause of potential error
        root_causes = analyzer.identify_root_cause(
            error_variable='result',
            causal_result=result
        )

        assert len(root_causes) > 0


class TestEpisodicMemory:
    """Test episodic memory system"""

    def test_priority_sampling(self):
        """Test priority-based sampling with ReplayStore"""
        from nerion_digital_physicist.infrastructure.memory import ReplayStore, Experience

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            replay_store = ReplayStore(storage_path)

            # Add test experiences
            for i in range(10):
                replay_store.append(
                    task_id=f"task_{i}",
                    template_id="test",
                    status="solved",
                    priority=float(i % 3),  # 0, 1, or 2
                    metadata={'test_data': True}
                )

            # Sample with priority
            samples = replay_store.sample(k=5, strategy='priority')

            assert len(samples) <= 5
            assert all(isinstance(s, Experience) for s in samples)


class TestModelRegistry:
    """Test model registry versioning"""

    def test_model_versioning(self):
        """Test model version management"""
        from nerion_digital_physicist.deployment.model_registry import ModelRegistry
        from nerion_digital_physicist.agent.brain import build_gnn

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry"
            registry_path.mkdir()

            registry = ModelRegistry(registry_path)

            # Create a dummy model
            model = build_gnn(
                architecture='sage',
                num_node_features=768,
                hidden_channels=256,
                num_classes=2
            )

            # Register the model
            model_version = registry.register(
                model=model,
                validation_accuracy=0.75,
                old_task_accuracy=0.70,
                architecture='sage'
            )

            assert model_version is not None
            assert '.' in model_version.version  # Should be semantic version
            assert model_version.validation_accuracy == 0.75


# Integration test for full learning cycle
@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_learning_cycle():
    """Test complete learning cycle from experience to model update"""
    from daemon.continuous_learner import ContinuousLearner
    from nerion_digital_physicist.infrastructure.memory import Experience

    with tempfile.TemporaryDirectory() as tmpdir:
        replay_root = Path(tmpdir) / "replay"
        curriculum_path = Path(tmpdir) / "curriculum.sqlite"
        model_registry_path = Path(tmpdir) / "models"

        replay_root.mkdir()
        model_registry_path.mkdir()

        # Create curriculum with test data
        import sqlite3
        conn = sqlite3.connect(str(curriculum_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY,
                name TEXT,
                before_code TEXT,
                after_code TEXT,
                test_code TEXT,
                language TEXT,
                difficulty TEXT,
                focus_area TEXT,
                category TEXT,
                tags TEXT
            )
        """)
        conn.execute("""
            INSERT INTO lessons (name, before_code, after_code, test_code, language, difficulty, focus_area, category, tags)
            VALUES ('test1', 'def bug(): return 1/0', 'def fixed(): return 1', 'def test(): pass', 'python', 'medium', 'A1', 'bug', '[]')
        """)
        conn.commit()
        conn.close()

        # Initialize learner
        learner = ContinuousLearner(
            replay_root=replay_root,
            curriculum_path=curriculum_path,
            model_registry_path=model_registry_path
        )

        # Note: This test verifies initialization, not full training
        # Full training would require more setup and time
        assert learner is not None
        assert learner.replay_store is not None
        assert learner.model_registry is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
