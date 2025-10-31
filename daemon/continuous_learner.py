"""
Continuous Learner for Daemon Integration

Autonomous continuous learning loop that runs in the daemon.

Integrates all components:
- MAML: Few-shot adaptation
- OnlineLearner: Incremental updates with EWC
- ProductionFeedbackCollector: Bug collection
- AutoCurriculumGenerator: Lesson generation
- ModelRegistry: Version management

Workflow:
1. Daemon detects production bugs → ProductionFeedbackCollector
2. When enough bugs accumulated → Trigger learning cycle
3. Sample high-priority bugs from ReplayStore
4. Generate lessons via AutoCurriculumGenerator
5. Update model incrementally via OnlineLearner
6. Validate and register new version via ModelRegistry
7. Deploy with canary testing

Usage in daemon:
    >>> # In daemon/nerion_daemon.py
    >>> async def train_gnn_background(self):
    ...     self.continuous_learner = ContinuousLearner(
    ...         replay_root=Path("data/replay"),
    ...         curriculum_path=Path("out/learning/curriculum.sqlite"),
    ...         model_registry_path=Path("out/models/registry")
    ...     )
    ...
    ...     while self.running:
    ...         if self.gnn_training:
    ...             updated = await self.continuous_learner.learning_cycle()
    ...             if updated:
    ...                 self.gnn_episodes += 1
    ...         await asyncio.sleep(3600)  # Every hour
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

# Import existing infrastructure (REUSE)
from nerion_digital_physicist.infrastructure.memory import ReplayStore, Experience
from nerion_digital_physicist.infrastructure.production_collector import (
    ProductionFeedbackCollector,
    ProductionBug
)
from nerion_digital_physicist.curriculum.auto_generator import (
    AutoCurriculumGenerator,
    store_lessons_in_curriculum
)
from nerion_digital_physicist.deployment.model_registry import (
    ModelRegistry,
    DeploymentStage
)
from nerion_digital_physicist.learning import LessonValidator

# Import new components
from nerion_digital_physicist.training.maml import MAMLTrainer, MAMLConfig
from nerion_digital_physicist.training.online_learner import (
    OnlineLearner,
    EWCConfig,
    ContinuousLearningEngine
)

# Import proper types
from nerion_digital_physicist.types import TrainingExample, TrainingBatch


@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning"""
    # Collection thresholds
    min_bugs_for_cycle: int = 50
    min_high_surprise_bugs: int = 10
    hours_between_cycles: int = 24

    # Learning parameters
    maml_enabled: bool = True
    ewc_lambda: float = 1000.0
    inner_learning_rate: float = 0.01
    outer_learning_rate: float = 0.001

    # Curriculum generation
    lessons_per_cycle: int = 30
    llm_provider: str = "gemini"

    # Validation
    min_validation_accuracy: float = 0.60
    max_forgetting_threshold: float = 0.10

    # Deployment
    canary_traffic_pct: float = 10.0
    canary_duration_hours: int = 24


@dataclass
class LearningCycleResult:
    """Result of a learning cycle"""
    success: bool
    new_version: Optional[str] = None
    bugs_processed: int = 0
    lessons_generated: int = 0
    validation_accuracy: float = 0.0
    old_task_accuracy: float = 0.0
    deployed_stage: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ContinuousLearner:
    """
    Autonomous continuous learning system for daemon.

    Coordinates all continuous learning components to enable
    autonomous improvement from production bugs.
    """

    def __init__(
        self,
        replay_root: Path,
        curriculum_path: Path,
        model_registry_path: Path,
        config: Optional[ContinuousLearningConfig] = None
    ):
        """
        Initialize continuous learner.

        Args:
            replay_root: Root directory for ReplayStore
            curriculum_path: Path to curriculum.sqlite
            model_registry_path: Path to model registry
            config: Continuous learning configuration
        """
        self.config = config or ContinuousLearningConfig()

        # Initialize components (REUSE existing infrastructure)
        self.replay_store = ReplayStore(replay_root)
        self.model_registry = ModelRegistry(model_registry_path)
        self.lesson_validator = LessonValidator()

        # Feedback collector (will set model later)
        self.feedback_collector = ProductionFeedbackCollector(
            replay_store=self.replay_store,
            model=None  # Set when model loaded
        )

        # Auto-curriculum generator
        self.auto_curriculum = AutoCurriculumGenerator(
            replay_store=self.replay_store,
            validator=self.lesson_validator,
            llm_provider=self.config.llm_provider
        )

        # Curriculum path
        self.curriculum_path = curriculum_path

        # Learning engine (will initialize with model)
        self.learning_engine: Optional[ContinuousLearningEngine] = None

        # Cycle history
        self.cycle_history: List[LearningCycleResult] = []

        print("[ContinuousLearner] Initialized")
        print(f"[ContinuousLearner] Config: "
              f"min_bugs={self.config.min_bugs_for_cycle}, "
              f"maml={self.config.maml_enabled}, "
              f"ewc_lambda={self.config.ewc_lambda}")

    async def learning_cycle(self) -> bool:
        """
        Run one complete autonomous learning cycle.

        Returns:
            True if model was updated, False otherwise
        """
        print("\n" + "="*60)
        print("[ContinuousLearner] Starting learning cycle")
        print("="*60)

        try:
            # Step 1: Check if should trigger cycle
            if not self._should_trigger_cycle():
                print("[ContinuousLearner] Insufficient bugs, skipping cycle")
                return False

            # Step 2: Load current production model
            current_model = self._load_current_model()
            if current_model is None:
                print("[ContinuousLearner] No production model found, skipping cycle")
                return False

            # Step 3: Sample high-priority production bugs
            high_priority_bugs = self.feedback_collector.get_high_surprise_bugs(
                k=self.config.min_bugs_for_cycle
            )
            print(f"[ContinuousLearner] Sampled {len(high_priority_bugs)} high-priority bugs")

            if len(high_priority_bugs) < self.config.min_bugs_for_cycle:
                print("[ContinuousLearner] Insufficient high-priority bugs")
                return False

            # Step 4: Generate lessons from production bugs
            lessons = self.auto_curriculum.generate_from_production(
                k=self.config.lessons_per_cycle
            )
            print(f"[ContinuousLearner] Generated {len(lessons)} lessons")

            if len(lessons) == 0:
                print("[ContinuousLearner] No lessons generated, skipping cycle")
                return False

            # Step 5: Store lessons in curriculum
            num_added, num_rejected = store_lessons_in_curriculum(
                lessons,
                self.curriculum_path
            )
            print(f"[ContinuousLearner] Stored {num_added} lessons in curriculum "
                  f"({num_rejected} duplicates rejected)")

            # Step 6: Convert bugs to training data
            new_training_data = self._prepare_training_data(high_priority_bugs)

            # Step 7: Sample diverse replay buffer (anti-forgetting)
            replay_buffer = self.replay_store.sample(k=200, strategy="random")
            replay_data = self._prepare_training_data(replay_buffer)

            # Step 8: Incremental update with MAML + EWC
            updated_model = await self._incremental_update(
                current_model=current_model,
                new_data=new_training_data,
                replay_data=replay_data
            )

            if updated_model is None:
                print("[ContinuousLearner] Incremental update failed")
                return False

            # Step 9: Validate updated model
            validation_metrics = await self._validate_model(
                updated_model,
                new_data=new_training_data,
                old_data=replay_data
            )

            if not self._meets_quality_threshold(validation_metrics):
                print("[ContinuousLearner] Model failed quality validation")
                return False

            # Step 10: Register new version
            # Extract hyperparameters from model architecture
            # (num_node_features from first graph in training data)
            first_graph = new_training_data[0][0] if new_training_data else None
            num_node_features = first_graph.num_node_features if first_graph else 768

            new_version = self.model_registry.register(
                model=updated_model,
                validation_accuracy=validation_metrics['validation_accuracy'],
                old_task_accuracy=validation_metrics['old_task_accuracy'],
                update_method="incremental",
                training_samples=len(new_training_data),
                notes=f"Learned from {len(high_priority_bugs)} production bugs",
                # Pass hyperparameters for proper model reconstruction
                architecture='sage',  # TODO: Make configurable
                num_node_features=num_node_features,
                hidden_channels=256,  # TODO: Extract from model if possible
                num_layers=4,         # TODO: Extract from model if possible
                residual=False,       # TODO: Extract from model if possible
                dropout=0.2,          # TODO: Extract from model if possible
                use_graphcodebert=False,  # TODO: Make configurable
                attention_heads=4     # TODO: Extract from model if possible
            )

            # Step 11: Canary deployment
            deployed = self.model_registry.promote(
                version=new_version.version,
                target_stage=DeploymentStage.CANARY,
                traffic_percentage=self.config.canary_traffic_pct
            )

            # Record cycle
            result = LearningCycleResult(
                success=True,
                new_version=new_version.version,
                bugs_processed=len(high_priority_bugs),
                lessons_generated=len(lessons),
                validation_accuracy=validation_metrics['validation_accuracy'],
                old_task_accuracy=validation_metrics['old_task_accuracy'],
                deployed_stage="canary" if deployed else "development"
            )
            self.cycle_history.append(result)

            # Reset metrics
            self.feedback_collector.reset_metrics()
            self.auto_curriculum.reset_metrics()

            print(f"[ContinuousLearner] Learning cycle complete!")
            print(f"[ContinuousLearner] New version: {new_version.version}")
            print(f"[ContinuousLearner] Validation accuracy: {validation_metrics['validation_accuracy']:.4f}")
            print(f"[ContinuousLearner] Old task accuracy: {validation_metrics['old_task_accuracy']:.4f}")
            print("="*60)

            return True

        except Exception as e:
            print(f"[ContinuousLearner] Error in learning cycle: {e}")
            import traceback
            traceback.print_exc()

            result = LearningCycleResult(
                success=False,
                error=str(e)
            )
            self.cycle_history.append(result)

            return False

    def collect_production_bug(
        self,
        bug: ProductionBug,
        graph: Any,
        ground_truth: Optional[int] = None
    ) -> float:
        """
        Collect a production bug.

        Called by daemon when bug is detected.

        Args:
            bug: Production bug information
            graph: AST graph representation
            ground_truth: Actual quality label

        Returns:
            Surprise score
        """
        return self.feedback_collector.collect_bug(bug, graph, ground_truth)

    def _should_trigger_cycle(self) -> bool:
        """Check if should trigger learning cycle"""
        return self.feedback_collector.should_trigger_learning_cycle()

    def _load_current_model(self) -> Optional[nn.Module]:
        """
        Load current production model from registry.

        Registry now returns fully constructed model with correct hyperparameters
        stored during registration.
        """
        try:
            # Registry now returns fully constructed model
            # (handles architecture + hyperparameters internally)
            model = self.model_registry.get_production_model()

            if model is None:
                print("[ContinuousLearner] No production model found in registry")
                return None

            print("[ContinuousLearner] Loaded production model successfully")
            return model

        except Exception as e:
            print(f"[ContinuousLearner] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_training_data(
        self,
        experiences: List[Experience]
    ) -> TrainingBatch:
        """
        Convert experiences to training data format.

        Args:
            experiences: List of Experience objects

        Returns:
            List of (graph, label) tuples
        """
        from nerion_digital_physicist.utils.graph_loader import GraphLoader

        # Initialize graph loader
        loader = GraphLoader(
            use_semantic_embeddings=True,
            cache_graphs=True
        )

        # Load graphs from experiences
        data = loader.load_from_experiences(experiences)

        return data

    async def _incremental_update(
        self,
        current_model: nn.Module,
        new_data: TrainingBatch,
        replay_data: TrainingBatch
    ) -> Optional[nn.Module]:
        """
        Incrementally update model with new data.

        Uses ContinuousLearningEngine (MAML + EWC).

        Args:
            current_model: Current model
            new_data: New training data
            replay_data: Replay buffer data

        Returns:
            Updated model or None if failed
        """
        print("[ContinuousLearner] Starting incremental update")

        # Initialize learning engine if needed
        if self.learning_engine is None:
            maml_config = MAMLConfig(
                inner_lr=self.config.inner_learning_rate,
                outer_lr=self.config.outer_learning_rate
            )
            ewc_config = EWCConfig(
                ewc_lambda=self.config.ewc_lambda
            )
            self.learning_engine = ContinuousLearningEngine(
                base_model=current_model,
                maml_config=maml_config,
                ewc_config=ewc_config
            )

        # Learn from production
        updated_model = self.learning_engine.learn_from_production(
            current_model=current_model,
            new_bugs=new_data,
            replay_buffer=replay_data,
            validation_set=replay_data[:100]  # Use subset for validation
        )

        return updated_model

    async def _validate_model(
        self,
        model: nn.Module,
        new_data: TrainingBatch,
        old_data: TrainingBatch
    ) -> Dict[str, float]:
        """
        Validate model on new and old tasks.

        Args:
            model: Model to validate
            new_data: New task data
            old_data: Old task data (forgetting metric)

        Returns:
            Validation metrics
        """
        print("[ContinuousLearner] Validating model")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # Validate on new data
        new_correct = 0
        new_total = 0
        with torch.no_grad():
            for graph, label in new_data[:100]:  # Sample to avoid OOM
                try:
                    # Move graph to device
                    graph = graph.to(device)

                    # Create batch tensor for single graph
                    # (indicates which graph each node belongs to)
                    batch_tensor = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

                    # Call model with proper PyG signature
                    out = model(
                        x=graph.x,
                        edge_index=graph.edge_index,
                        batch=batch_tensor
                    )  # Shape: [1, num_classes]

                    pred = out.argmax(dim=-1).item()  # Get class prediction
                    new_correct += (pred == label)
                    new_total += 1
                except Exception as e:
                    print(f"[ContinuousLearner] Validation error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        new_acc = new_correct / new_total if new_total > 0 else 0.0

        # Validate on old data (forgetting metric)
        old_correct = 0
        old_total = 0
        with torch.no_grad():
            for graph, label in old_data[:100]:  # Sample
                try:
                    # Move graph to device
                    graph = graph.to(device)

                    # Create batch tensor for single graph
                    batch_tensor = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

                    # Call model with proper PyG signature
                    out = model(
                        x=graph.x,
                        edge_index=graph.edge_index,
                        batch=batch_tensor
                    )  # Shape: [1, num_classes]

                    pred = out.argmax(dim=-1).item()  # Get class prediction
                    old_correct += (pred == label)
                    old_total += 1
                except Exception as e:
                    continue

        old_acc = old_correct / old_total if old_total > 0 else 0.0

        # Calculate forgetting
        forgetting = max(0, old_acc - new_acc) if old_total > 0 else 0.0

        print(f"[ContinuousLearner] Validation: new_acc={new_acc:.3f}, old_acc={old_acc:.3f}, forgetting={forgetting:.3f}")

        return {
            'validation_accuracy': new_acc,
            'old_task_accuracy': old_acc,
            'forgetting': forgetting
        }

    def _meets_quality_threshold(self, metrics: Dict[str, float]) -> bool:
        """Check if model meets quality thresholds"""
        if metrics['validation_accuracy'] < self.config.min_validation_accuracy:
            print(f"[ContinuousLearner] Validation accuracy too low: "
                  f"{metrics['validation_accuracy']:.4f} < {self.config.min_validation_accuracy}")
            return False

        if metrics['forgetting'] > self.config.max_forgetting_threshold:
            print(f"[ContinuousLearner] Too much forgetting: "
                  f"{metrics['forgetting']:.4f} > {self.config.max_forgetting_threshold}")
            return False

        return True

    def get_cycle_history(self) -> List[LearningCycleResult]:
        """Get history of learning cycles"""
        return self.cycle_history

    def get_metrics(self) -> Dict[str, Any]:
        """Get current continuous learning metrics"""
        return {
            'feedback_collector': self.feedback_collector.get_metrics(),
            'auto_curriculum': self.auto_curriculum.get_metrics(),
            'cycle_history': [
                {
                    'timestamp': result.timestamp,
                    'success': result.success,
                    'version': result.new_version,
                    'bugs_processed': result.bugs_processed,
                    'lessons_generated': result.lessons_generated,
                    'validation_accuracy': result.validation_accuracy,
                }
                for result in self.cycle_history[-10:]  # Last 10 cycles
            ]
        }


# Example daemon integration
async def daemon_example():
    """Example of daemon integration"""
    from pathlib import Path

    # Initialize continuous learner
    learner = ContinuousLearner(
        replay_root=Path("data/replay"),
        curriculum_path=Path("out/learning/curriculum.sqlite"),
        model_registry_path=Path("out/models/registry")
    )

    # Simulate daemon loop
    print("=== Daemon Continuous Learning Loop ===")

    while True:
        # Run learning cycle every hour
        updated = await learner.learning_cycle()

        if updated:
            print("[Daemon] Model updated! New version deployed to canary.")

        # Wait 1 hour
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(daemon_example())
