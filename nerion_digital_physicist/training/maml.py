"""
MAML (Model-Agnostic Meta-Learning) for Few-Shot Adaptation

Enables the GNN to rapidly adapt to new bug patterns with minimal examples.
Based on "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (Finn et al., 2017)

Key Concepts:
- Inner loop: Fast adaptation on support set (few examples)
- Outer loop: Meta-optimization across tasks
- Second-order gradients: Learn initialization that adapts quickly

Integration with Nerion:
- Each production bug becomes a "task"
- Support set: 1-5 similar bug examples
- Query set: New bug to classify
- Meta-learns initialization that generalizes across bug types
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
class MAMLConfig:
    """Configuration for MAML training"""
    inner_lr: float = 0.01           # Learning rate for inner loop (task adaptation)
    outer_lr: float = 0.001          # Learning rate for outer loop (meta-optimization)
    inner_steps: int = 5             # Gradient steps per task in inner loop
    meta_batch_size: int = 8         # Number of tasks per meta-batch
    support_size: int = 3            # Examples per task for adaptation
    query_size: int = 5              # Examples per task for evaluation
    first_order: bool = False        # Use first-order approximation (FOMAML)

    # Regularization
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Convergence
    min_improvement: float = 0.001   # Minimum meta-loss improvement
    patience: int = 3                # Early stopping patience


@dataclass
class MAMLTask:
    """A single meta-learning task (e.g., one bug type)"""
    task_id: str
    support_graphs: List[Any]        # Support set graphs (for adaptation)
    support_labels: List[int]        # Support set labels
    query_graphs: List[Any]          # Query set graphs (for evaluation)
    query_labels: List[int]          # Query set labels
    metadata: Dict[str, Any]         # Bug type, context, etc.


class MAMLTrainer:
    """
    MAML trainer for few-shot GNN adaptation.

    Usage:
        >>> config = MAMLConfig(inner_lr=0.01, inner_steps=5)
        >>> maml = MAMLTrainer(base_model, config)
        >>> tasks = create_tasks_from_bugs(production_bugs)
        >>> adapted_model = maml.meta_train(tasks, epochs=100)
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[MAMLConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize MAML trainer.

        Args:
            base_model: Base GNN model to meta-train
            config: MAML configuration
            device: Device for training (auto-detect if None)
        """
        self.config = config or MAMLConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Clone base model for meta-training
        self.meta_model = copy.deepcopy(base_model).to(self.device)
        self.meta_optimizer = Adam(
            self.meta_model.parameters(),
            lr=self.config.outer_lr,
            weight_decay=self.config.weight_decay
        )

        # Track meta-learning progress
        self.meta_loss_history: List[float] = []
        self.meta_accuracy_history: List[float] = []

    def inner_loop(
        self,
        task: MAMLTask,
        model: nn.Module,
    ) -> Tuple[nn.Module, float]:
        """
        Inner loop: Fast adaptation on support set.

        Args:
            task: Task with support/query sets
            model: Model to adapt (will be cloned)

        Returns:
            (adapted_model, support_loss)
        """
        # Clone model for task-specific adaptation
        adapted_model = copy.deepcopy(model)
        inner_optimizer = Adam(
            adapted_model.parameters(),
            lr=self.config.inner_lr
        )

        # Create loader for support set
        support_loader = self._create_loader(task.support_graphs, task.support_labels)

        # Adapt on support set
        adapted_model.train()
        final_loss = 0.0
        
        for step in range(self.config.inner_steps):
            step_loss = 0.0
            num_batches = 0
            
            for batch in support_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                # Extract GraphCodeBERT embeddings if available
                graphcodebert_emb = None
                if hasattr(batch, 'graphcodebert_embedding'):
                    num_graphs = batch.num_graphs
                    graphcodebert_emb = batch.graphcodebert_embedding.view(num_graphs, -1)
                
                edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None

                logits = adapted_model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    graphcodebert_embedding=graphcodebert_emb,
                    edge_attr=edge_attr
                )
                
                labels = batch.y.view(-1)

                # Compute loss and adapt
                loss = F.cross_entropy(logits, labels)
                inner_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    adapted_model.parameters(),
                    self.config.grad_clip
                )

                inner_optimizer.step()
                
                step_loss += loss.item()
                num_batches += 1
            
            final_loss = step_loss / max(1, num_batches)

        return adapted_model, final_loss

    def outer_loop_step(
        self,
        tasks: List[MAMLTask]
    ) -> Tuple[float, float]:
        """
        Outer loop: Meta-optimization across tasks.

        Args:
            tasks: Batch of tasks for meta-update

        Returns:
            (meta_loss, meta_accuracy)
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        meta_correct = 0
        meta_total = 0

        for task in tasks:
            # Inner loop: Adapt to task
            if self.config.first_order:
                # FOMAML: Don't compute second-order gradients
                with torch.no_grad():
                    adapted_model, _ = self.inner_loop(task, self.meta_model)
            else:
                # Full MAML: Compute second-order gradients
                adapted_model, _ = self.inner_loop(task, self.meta_model)

            # Evaluate on query set
            adapted_model.eval()
            
            # Create loader for query set
            query_loader = self._create_loader(task.query_graphs, task.query_labels)
            
            with torch.set_grad_enabled(not self.config.first_order):
                task_loss = 0.0
                num_batches = 0
                
                for batch in query_loader:
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    graphcodebert_emb = None
                    if hasattr(batch, 'graphcodebert_embedding'):
                        num_graphs = batch.num_graphs
                        graphcodebert_emb = batch.graphcodebert_embedding.view(num_graphs, -1)
                    
                    edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None

                    logits = adapted_model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        graphcodebert_embedding=graphcodebert_emb,
                        edge_attr=edge_attr
                    )
                    
                    labels = batch.y.view(-1)

                    # Compute query loss (meta-objective)
                    loss = F.cross_entropy(logits, labels)
                    task_loss += loss

                    # Track accuracy
                    predictions = logits.argmax(dim=1)
                    meta_correct += (predictions == labels).sum().item()
                    meta_total += labels.size(0)
                    num_batches += 1
                
                # Average loss over batches if needed, but usually query set is small enough for one batch
                # Here we sum losses for backprop
                meta_loss += task_loss

        # Average across tasks
        meta_loss = meta_loss / len(tasks)
        meta_accuracy = meta_correct / meta_total if meta_total > 0 else 0.0

        # Meta-optimization
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.meta_model.parameters(),
            self.config.grad_clip
        )
        self.meta_optimizer.step()

        return meta_loss.item(), meta_accuracy

    def _create_loader(self, graphs: List[Any], labels: List[int]) -> Any:
        """Create PyG DataLoader from graphs and labels"""
        from torch_geometric.loader import DataLoader
        from torch_geometric.data import Data
        
        processed_data = []
        for graph, label in zip(graphs, labels):
            if not isinstance(graph, Data):
                continue
            d = graph.clone()
            d.y = torch.tensor([label], dtype=torch.long)
            processed_data.append(d)
            
        # Use a batch size large enough to hold all support/query examples if possible
        # or default to something reasonable like 32
        return DataLoader(
            processed_data,
            batch_size=32,
            shuffle=True
        )

    def meta_train(
        self,
        tasks: List[MAMLTask],
        epochs: int = 100,
        validation_tasks: Optional[List[MAMLTask]] = None
    ) -> nn.Module:
        """
        Meta-train the model across tasks.

        Args:
            tasks: Training tasks
            epochs: Number of meta-training epochs
            validation_tasks: Optional validation tasks

        Returns:
            Meta-trained model
        """
        print(f"[MAML] Starting meta-training with {len(tasks)} tasks")
        print(f"[MAML] Config: inner_lr={self.config.inner_lr}, "
              f"inner_steps={self.config.inner_steps}, "
              f"meta_batch_size={self.config.meta_batch_size}")

        best_val_accuracy = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # Sample meta-batch of tasks
            import random
            batch_tasks = random.sample(
                tasks,
                min(self.config.meta_batch_size, len(tasks))
            )

            # Meta-optimization step
            meta_loss, meta_accuracy = self.outer_loop_step(batch_tasks)

            self.meta_loss_history.append(meta_loss)
            self.meta_accuracy_history.append(meta_accuracy)

            # Validation
            val_accuracy = 0.0
            if validation_tasks and epoch % 10 == 0:
                val_accuracy = self.evaluate(validation_tasks)

                # Early stopping
                if val_accuracy > best_val_accuracy + self.config.min_improvement:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    print(f"[MAML] Early stopping at epoch {epoch}")
                    break

            # Logging
            if epoch % 10 == 0:
                print(f"[MAML] Epoch {epoch}/{epochs}: "
                      f"meta_loss={meta_loss:.4f}, "
                      f"meta_accuracy={meta_accuracy:.4f}, "
                      f"val_accuracy={val_accuracy:.4f}")

        print(f"[MAML] Meta-training complete. Best val accuracy: {best_val_accuracy:.4f}")
        return self.meta_model

    def evaluate(self, tasks: List[MAMLTask]) -> float:
        """
        Evaluate meta-trained model on tasks.

        Args:
            tasks: Evaluation tasks

        Returns:
            Average accuracy across tasks
        """
        self.meta_model.eval()
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for task in tasks:
                # Adapt to task
                adapted_model, _ = self.inner_loop(task, self.meta_model)
                adapted_model.eval()

                # Create loader for query set
                query_loader = self._create_loader(task.query_graphs, task.query_labels)
                
                for batch in query_loader:
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    graphcodebert_emb = None
                    if hasattr(batch, 'graphcodebert_embedding'):
                        num_graphs = batch.num_graphs
                        graphcodebert_emb = batch.graphcodebert_embedding.view(num_graphs, -1)
                    
                    edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None

                    logits = adapted_model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        graphcodebert_embedding=graphcodebert_emb,
                        edge_attr=edge_attr
                    )
                    
                    labels = batch.y.view(-1)
                    predictions = logits.argmax(dim=1)
                    total_correct += (predictions == labels).sum().item()
                    total_examples += labels.size(0)

        return total_correct / total_examples if total_examples > 0 else 0.0

    def adapt_to_new_task(
        self,
        support_examples: List[Tuple[Any, int]],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt meta-trained model to a new task (production deployment).

        Args:
            support_examples: List of (graph, label) tuples
            num_steps: Number of adaptation steps (uses config default if None)

        Returns:
            Adapted model ready for inference
        """
        # Create task from support examples
        support_graphs, support_labels = zip(*support_examples)
        task = MAMLTask(
            task_id="production",
            support_graphs=list(support_graphs),
            support_labels=list(support_labels),
            query_graphs=[],  # Not used for adaptation-only
            query_labels=[],
            metadata={}
        )

        # Override inner steps if specified
        if num_steps is not None:
            original_steps = self.config.inner_steps
            self.config.inner_steps = num_steps

        # Adapt model
        adapted_model, _ = self.inner_loop(task, self.meta_model)

        # Restore config
        if num_steps is not None:
            self.config.inner_steps = original_steps

        return adapted_model

    def save_checkpoint(self, path: Path):
        """Save MAML checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'meta_model_state': self.meta_model.state_dict(),
            'meta_optimizer_state': self.meta_optimizer.state_dict(),
            'config': self.config,
            'meta_loss_history': self.meta_loss_history,
            'meta_accuracy_history': self.meta_accuracy_history,
        }, path)
        print(f"[MAML] Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path):
        """Load MAML checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_model.load_state_dict(checkpoint['meta_model_state'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state'])
        self.config = checkpoint['config']
        self.meta_loss_history = checkpoint['meta_loss_history']
        self.meta_accuracy_history = checkpoint['meta_accuracy_history']
        print(f"[MAML] Checkpoint loaded: {path}")


def create_tasks_from_curriculum(
    curriculum_lessons: List[Dict[str, Any]],
    support_size: int = 3,
    query_size: int = 5
) -> List[MAMLTask]:
    """
    Convert curriculum lessons into MAML tasks.

    Each bug type becomes a task with support/query split.

    Args:
        curriculum_lessons: Lessons from curriculum.sqlite
        support_size: Examples per task for adaptation
        query_size: Examples per task for evaluation

    Returns:
        List of MAML tasks
    """
    from collections import defaultdict

    # Group lessons by bug type
    bug_type_lessons = defaultdict(list)
    for lesson in curriculum_lessons:
        bug_type = lesson.get('category', 'unknown')
        bug_type_lessons[bug_type].append(lesson)

    tasks = []
    for bug_type, lessons in bug_type_lessons.items():
        if len(lessons) < support_size + query_size:
            continue  # Not enough examples for this bug type

        # Split into support and query
        import random
        random.shuffle(lessons)
        support_lessons = lessons[:support_size]
        query_lessons = lessons[support_size:support_size + query_size]

        # TODO: Convert lessons to graphs (requires dataset_builder integration)
        # For now, create placeholder task structure
        task = MAMLTask(
            task_id=f"bug_type_{bug_type}",
            support_graphs=[],  # Populate with actual graphs
            support_labels=[],
            query_graphs=[],
            query_labels=[],
            metadata={'bug_type': bug_type, 'num_lessons': len(lessons)}
        )
        tasks.append(task)

    return tasks
