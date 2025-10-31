"""
Online Learning Engine with EWC (Elastic Weight Consolidation)

Enables continuous learning from production bugs without catastrophic forgetting.
Based on "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)

Key Concepts:
- EWC: Penalizes changes to important weights (Fisher Information Matrix)
- Incremental updates: Learn from new data without forgetting old
- Experience replay: Mix old and new data during training
- Adaptive learning rates: Slow down learning on important weights

Integration with Nerion:
- Daemon triggers incremental updates when new bugs accumulate
- ReplayStore provides diverse old experiences for anti-forgetting
- Model Registry tracks versions and enables rollback
- Production feedback provides surprise-weighted samples
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


@dataclass
class EWCConfig:
    """Configuration for EWC"""
    fisher_samples: int = 200         # Samples for Fisher computation
    ewc_lambda: float = 1000.0        # EWC regularization strength
    online_ewc: bool = True           # Use online EWC (cumulative Fisher)
    gamma: float = 0.9                # Decay factor for online EWC

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Experience replay
    replay_ratio: float = 0.5         # Ratio of old experiences in batch
    min_replay_samples: int = 100     # Minimum replay buffer size


@dataclass
class IncrementalUpdate:
    """Result of an incremental learning update"""
    success: bool
    new_accuracy: float
    old_accuracy: float               # Accuracy on old tasks (forgetting metric)
    ewc_loss: float
    task_loss: float
    num_new_samples: int
    num_replay_samples: int
    model_version: str


class OnlineLearner:
    """
    Online learning engine with EWC for continuous learning.

    Usage:
        >>> config = EWCConfig(ewc_lambda=1000.0)
        >>> learner = OnlineLearner(base_model, config)
        >>>
        >>> # When new production bugs accumulate
        >>> updated_model = learner.incremental_update(
        ...     current_model=model,
        ...     new_data=production_bugs,
        ...     replay_data=diverse_old_samples
        ... )
    """

    def __init__(
        self,
        config: Optional[EWCConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize online learner.

        Args:
            config: EWC configuration
            device: Device for training
        """
        self.config = config or EWCConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # EWC parameters (cumulative across tasks)
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0

        # Training history
        self.update_history: List[IncrementalUpdate] = []

    def compute_fisher_information(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal.

        Fisher diagonal approximates parameter importance:
        F_i = E[(∂log p(y|x,θ)/∂θ_i)²]

        Args:
            model: Model to compute Fisher for
            data_loader: Data to estimate Fisher from

        Returns:
            Dictionary mapping parameter name to Fisher diagonal
        """
        model.eval()
        fisher = {}

        # Initialize Fisher accumulators
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        # Accumulate gradients squared
        num_samples = 0
        for batch_idx, (graphs, labels) in enumerate(data_loader):
            if batch_idx >= self.config.fisher_samples // self.config.batch_size:
                break

            model.zero_grad()

            # Forward pass
            logits = []
            for graph in graphs:
                logit = model(graph.to(self.device))
                logits.append(logit)
            logits = torch.stack(logits)
            labels = labels.to(self.device)

            # Sample from model's distribution (not ground truth)
            predictions = F.softmax(logits, dim=1)
            sampled_labels = torch.multinomial(predictions, 1).squeeze()

            # Compute log likelihood
            loss = F.cross_entropy(logits, sampled_labels)
            loss.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            num_samples += len(graphs)

        # Average over samples
        for name in fisher:
            fisher[name] /= num_samples

        return fisher

    def consolidate_task(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ):
        """
        Consolidate current task by computing Fisher and saving parameters.

        Args:
            model: Model after training on current task
            data_loader: Data from current task
        """
        print(f"[EWC] Consolidating task {self.task_count + 1}")

        # Compute Fisher for current task
        current_fisher = self.compute_fisher_information(model, data_loader)

        # Update cumulative Fisher (online EWC)
        if self.config.online_ewc and self.task_count > 0:
            for name in current_fisher:
                if name in self.fisher_dict:
                    # Exponential moving average
                    self.fisher_dict[name] = (
                        self.config.gamma * self.fisher_dict[name] +
                        current_fisher[name]
                    )
                else:
                    self.fisher_dict[name] = current_fisher[name]
        else:
            # First task or standard EWC
            for name in current_fisher:
                self.fisher_dict[name] = current_fisher[name]

        # Save optimal parameters for current task
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

        self.task_count += 1
        print(f"[EWC] Task consolidated. Total tasks: {self.task_count}")

    def ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularization penalty.

        Penalty = λ/2 * Σ_i F_i * (θ_i - θ*_i)²

        Args:
            model: Current model

        Returns:
            EWC penalty (scalar tensor)
        """
        if self.task_count == 0:
            return torch.tensor(0.0, device=self.device)

        penalty = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_dict and name in self.optimal_params:
                penalty += (
                    self.fisher_dict[name] *
                    (param - self.optimal_params[name]).pow(2)
                ).sum()

        return self.config.ewc_lambda / 2 * penalty

    def incremental_update(
        self,
        current_model: nn.Module,
        new_data: List[Tuple[Any, int]],
        replay_data: Optional[List[Tuple[Any, int]]] = None,
        validation_data: Optional[List[Tuple[Any, int]]] = None
    ) -> Tuple[nn.Module, IncrementalUpdate]:
        """
        Incrementally update model with new data while preserving old knowledge.

        Args:
            current_model: Current production model
            new_data: New production bug examples (graph, label)
            replay_data: Old experiences for anti-forgetting
            validation_data: Validation set for old tasks (forgetting metric)

        Returns:
            (updated_model, update_info)
        """
        print(f"[OnlineLearner] Starting incremental update")
        print(f"[OnlineLearner] New samples: {len(new_data)}")
        print(f"[OnlineLearner] Replay samples: {len(replay_data) if replay_data else 0}")

        # Clone model for training
        model = copy.deepcopy(current_model).to(self.device)
        model.train()

        optimizer = Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Prepare mixed dataset (new + replay)
        if replay_data and len(replay_data) >= self.config.min_replay_samples:
            # Mix new and replay data
            num_replay = int(len(new_data) * self.config.replay_ratio)
            import random
            replay_sample = random.sample(replay_data, min(num_replay, len(replay_data)))
            mixed_data = new_data + replay_sample
            random.shuffle(mixed_data)
        else:
            mixed_data = new_data

        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            torch.arange(len(mixed_data))  # Placeholder indices
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Training loop
        total_task_loss = 0.0
        total_ewc_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.num_epochs):
            epoch_task_loss = 0.0
            epoch_ewc_loss = 0.0

            for batch_indices in data_loader:
                optimizer.zero_grad()

                # Get batch data
                batch_data = [mixed_data[i] for i in batch_indices[0]]
                graphs, labels = zip(*batch_data)

                # Forward pass
                logits = []
                for graph in graphs:
                    logit = model(graph.to(self.device))
                    logits.append(logit)
                logits = torch.stack(logits)
                labels = torch.tensor(labels, dtype=torch.long, device=self.device)

                # Task loss
                task_loss = F.cross_entropy(logits, labels)

                # EWC penalty
                ewc_loss = self.ewc_penalty(model)

                # Total loss
                total_loss = task_loss + ewc_loss

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.grad_clip
                )
                optimizer.step()

                epoch_task_loss += task_loss.item()
                epoch_ewc_loss += ewc_loss.item()
                num_batches += 1

            # Logging
            avg_task_loss = epoch_task_loss / len(data_loader)
            avg_ewc_loss = epoch_ewc_loss / len(data_loader)
            print(f"[OnlineLearner] Epoch {epoch + 1}/{self.config.num_epochs}: "
                  f"task_loss={avg_task_loss:.4f}, ewc_loss={avg_ewc_loss:.4f}")

        total_task_loss = epoch_task_loss / len(data_loader)
        total_ewc_loss = epoch_ewc_loss / len(data_loader)

        # Evaluate on new data
        new_accuracy = self._evaluate(model, new_data)

        # Evaluate on old data (forgetting metric)
        old_accuracy = 0.0
        if validation_data:
            old_accuracy = self._evaluate(model, validation_data)

        # Consolidate this task
        new_data_loader = self._create_data_loader(new_data)
        self.consolidate_task(model, new_data_loader)

        # Record update
        update_info = IncrementalUpdate(
            success=True,
            new_accuracy=new_accuracy,
            old_accuracy=old_accuracy,
            ewc_loss=total_ewc_loss,
            task_loss=total_task_loss,
            num_new_samples=len(new_data),
            num_replay_samples=len(replay_data) if replay_data else 0,
            model_version=f"v{self.task_count}"
        )
        self.update_history.append(update_info)

        print(f"[OnlineLearner] Update complete: new_acc={new_accuracy:.4f}, "
              f"old_acc={old_accuracy:.4f}")

        return model, update_info

    def _evaluate(
        self,
        model: nn.Module,
        data: List[Tuple[Any, int]]
    ) -> float:
        """Evaluate model accuracy on data"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for graph, label in data:
                logit = model(graph.to(self.device))
                prediction = logit.argmax(dim=0).item()
                correct += (prediction == label)
                total += 1

        return correct / total if total > 0 else 0.0

    def _create_data_loader(
        self,
        data: List[Tuple[Any, int]]
    ) -> torch.utils.data.DataLoader:
        """Create data loader from (graph, label) list"""
        dataset = torch.utils.data.TensorDataset(
            torch.arange(len(data))
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

    def save_checkpoint(self, path: Path):
        """Save EWC state"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'fisher_dict': self.fisher_dict,
            'optimal_params': self.optimal_params,
            'task_count': self.task_count,
            'config': self.config,
            'update_history': self.update_history,
        }, path)
        print(f"[OnlineLearner] Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path):
        """Load EWC state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.fisher_dict = checkpoint['fisher_dict']
        self.optimal_params = checkpoint['optimal_params']
        self.task_count = checkpoint['task_count']
        self.config = checkpoint['config']
        self.update_history = checkpoint.get('update_history', [])
        print(f"[OnlineLearner] Checkpoint loaded: {path}")
        print(f"[OnlineLearner] Consolidated tasks: {self.task_count}")


class ContinuousLearningEngine:
    """
    High-level continuous learning engine combining MAML + EWC + Replay.

    Integrates with Nerion's existing infrastructure:
    - ReplayStore for experience storage
    - MAMLTrainer for few-shot adaptation
    - OnlineLearner for incremental updates
    """

    def __init__(
        self,
        base_model: nn.Module,
        maml_config: Optional[Any] = None,
        ewc_config: Optional[EWCConfig] = None
    ):
        """
        Initialize continuous learning engine.

        Args:
            base_model: Base GNN model
            maml_config: MAML configuration
            ewc_config: EWC configuration
        """
        # Import here to avoid circular dependency
        from nerion_digital_physicist.training.maml import MAMLTrainer, MAMLConfig

        self.maml = MAMLTrainer(base_model, maml_config or MAMLConfig())
        self.online_learner = OnlineLearner(ewc_config)
        self.base_model = base_model

    def learn_from_production(
        self,
        current_model: nn.Module,
        new_bugs: List[Tuple[Any, int]],
        replay_buffer: List[Tuple[Any, int]],
        validation_set: Optional[List[Tuple[Any, int]]] = None
    ) -> nn.Module:
        """
        Learn from production bugs using MAML + EWC.

        Workflow:
        1. Use MAML to adapt to new bug pattern (few-shot)
        2. Use EWC to incrementally update without forgetting
        3. Mix with replay buffer for anti-forgetting

        Args:
            current_model: Current production model
            new_bugs: New production bug examples
            replay_buffer: Diverse old experiences
            validation_set: Old task validation for forgetting metric

        Returns:
            Updated model
        """
        print(f"[ContinuousLearning] Learning from {len(new_bugs)} production bugs")

        # Step 1: MAML adaptation (if few examples)
        if len(new_bugs) <= 10:
            print(f"[ContinuousLearning] Few-shot scenario, using MAML adaptation")
            adapted_model = self.maml.adapt_to_new_task(
                support_examples=new_bugs[:5],  # Use subset for adaptation
                num_steps=5
            )
        else:
            adapted_model = current_model

        # Step 2: EWC incremental update with replay
        updated_model, update_info = self.online_learner.incremental_update(
            current_model=adapted_model,
            new_data=new_bugs,
            replay_data=replay_buffer,
            validation_data=validation_set
        )

        print(f"[ContinuousLearning] Update complete: "
              f"new_acc={update_info.new_accuracy:.4f}, "
              f"old_acc={update_info.old_accuracy:.4f}")

        return updated_model
