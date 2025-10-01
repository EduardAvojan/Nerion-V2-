"""Phase 2 agent: integrates curiosity-driven planning with scope-aware edits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch_geometric.data import Data

from .brain import CodeGraphNN
from .semantics import SemanticEmbedder
from ..environment.actions import Action
from ..environment.refactor_env import RenameAction, TestOutcome


def _ensure_batch(graph: Data) -> Data:
    """Guarantee that a graph has a batch vector for pooling operations."""
    if getattr(graph, "batch", None) is None:
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    return graph


@dataclass
class EpisodeResult:
    """Summary of a single agent episode for downstream logging."""

    action: Action
    predicted_pass: float
    predicted_fail: float
    outcome_is_success: bool
    surprise: float
    memory_size: int
    policy_mode: str
    policy_epsilon: float
    policy_uncertainty: float
    policy_entropy: float
    policy_entropy_bonus: float
    policy_visit_count: int
    policy_epsilon_next: float
    action_tags: tuple[str, ...]
    action_metadata: dict[str, object]


@dataclass
class PolicyDecision:
    """Decision metadata describing how the action was selected."""

    action: Action
    mode: str  # "epsilon" or "curiosity"
    epsilon: float
    uncertainty: float
    entropy: float
    visit_count: int


class AgentV2:
    """Agent that perceives, imagines, acts, and learns using a GNN brain."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        *,
        epsilon: float = 0.1,
        epsilon_min: float = 0.0,
        epsilon_max: Optional[float] = None,
        epsilon_decay: float = 1.0,
        epsilon_step: float = 0.0,
        adaptive_surprise_target: float = 0.0,
        adaptive_epsilon: bool = False,
        entropy_bonus: float = 0.0,
        embedder: Optional[SemanticEmbedder] = None,
        num_mc_passes: int = 10,
    ):
        if input_dim is None:
            raise ValueError("AgentV2 requires a specific `input_dim`.")

        self.brain = CodeGraphNN(
            num_node_features=input_dim,
            hidden_channels=hidden_dim,
            num_classes=1,  # Output a single continuous value (the predicted success score)
        )
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=learning_rate)
        # Use Mean Squared Error for regressing towards a success score
        self.criterion = torch.nn.MSELoss()
        self.memory: list[tuple[Data, torch.Tensor]] = []
        self.epsilon = max(0.0, float(epsilon))
        self.epsilon_min = max(0.0, float(epsilon_min))
        if epsilon_max is not None:
            self.epsilon_max = max(self.epsilon_min, float(epsilon_max))
        else:
            self.epsilon_max = max(self.epsilon, self.epsilon_min)
        self.epsilon_decay = float(epsilon_decay) if epsilon_decay > 0 else 1.0
        self.epsilon_step = max(0.0, float(epsilon_step))
        self.adaptive_surprise_target = max(0.0, float(adaptive_surprise_target))
        self.adaptive_epsilon = bool(adaptive_epsilon)
        self.entropy_bonus = float(entropy_bonus)
        self.num_mc_passes = num_mc_passes
        self._checkpoint_metadata: dict[str, Any] = {}

    def predict_with_uncertainty(self, graph_data: Data, node_map: Dict[int, Dict], action: RenameAction) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the outcome of an action and estimates the model's uncertainty.
        Returns tensors for mean prediction and uncertainty.
        """
        predictions = []
        target_node_idx = self._find_node_index(action, node_map)
        if target_node_idx is None:
            # Return tensors to maintain type consistency
            return torch.tensor(0.0), torch.tensor(0.0)

        for _ in range(self.num_mc_passes):
            action_feature = torch.zeros(graph_data.num_nodes, 1)
            action_feature[target_node_idx] = 1.0
            augmented_x = torch.cat([graph_data.x, action_feature], dim=1)

            prediction_logits = self.brain(augmented_x, graph_data.edge_index, graph_data.batch, use_dropout=True)
            logit_for_action = prediction_logits[target_node_idx]
            predictions.append(torch.sigmoid(logit_for_action))

        predictions_tensor = torch.stack(predictions)
        mean_prediction = predictions_tensor.mean()
        uncertainty = predictions_tensor.var()
        return mean_prediction, uncertainty

    def learn(
        self, 
        experiences: List[Tuple[RenameAction, TestOutcome]], 
        graph_data: Data, 
        node_map: Dict[int, Dict]
    ) -> float:
        """Trains the brain on a batch of experiences from the refactoring environment."""
        self.brain.train()
        total_loss = 0

        self.optimizer.zero_grad()

        for action, outcome in experiences:
            mean_prediction, uncertainty = self.predict_with_uncertainty(graph_data, node_map, action)

            # Calculate a continuous success score (0.0 to 1.0)
            total_tests = outcome.passed + outcome.failed + outcome.errored
            if total_tests == 0:
                success_score = 0.0 # Or 1.0, depending on desired behavior for no tests
            else:
                success_score = outcome.passed / total_tests

            prediction_error = abs(mean_prediction.item() - success_score)
            surprise = prediction_error / (uncertainty.item() + 1e-6)
            self._update_adaptive_parameters(surprise)

            target = torch.tensor([success_score])
            loss = self.criterion(mean_prediction, target)
            total_loss += loss.item()

            loss.backward()

        self.optimizer.step()

        return total_loss / len(experiences)

    def _find_node_index(self, action: RenameAction, node_map: Dict[int, Dict]) -> Optional[int]:
        """Helper to find the graph node index corresponding to a rename action."""
        for idx, node_info in node_map.items():
            # The node_id in the map is structured like 'path/to/file.py::ClassName::method_name'
            node_id_parts = node_info.get("id", "").split("::")
            file_path = node_id_parts[0]
            
            # Match file path and the name of the function/class being renamed
            if file_path == action.file_path and node_info.get("name") == action.old_name:
                return idx
        return None

    def _update_adaptive_parameters(self, surprise: float) -> float:
        """Optionally adjust epsilon based on surprise observations."""

        if not self.adaptive_epsilon:
            return self.epsilon

        if surprise < self.adaptive_surprise_target:
            # Reduce exploration if we are confident.
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            if self.epsilon_step > 0:
                self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_step)

        return self.epsilon

    def save_brain(self, path: Path) -> None:
        """Persist the model (and optimizer) parameters to disk."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.brain.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "feature_dim": self.brain.conv1.in_channels,
        }
        checkpoint.update(self._checkpoint_metadata)
        torch.save(checkpoint, path)

    def load_brain(self, path: Path) -> bool:
        """Load model (and optimizer) parameters if the checkpoint exists."""

        path = Path(path)
        if not path.is_file():
            return False

        checkpoint = torch.load(path, map_location="cpu")
        model_state = checkpoint.get("model_state")
        if model_state is None:
            raise ValueError(f"Checkpoint at {path} missing 'model_state'")

        expected_dim = self.brain.conv1.in_channels
        feature_dim = checkpoint.get("feature_dim", expected_dim)
        if feature_dim != expected_dim:
            raise ValueError(
                "Checkpoint feature dimension mismatch: "
                f"expected {expected_dim}, found {feature_dim}"
            )

        self.brain.load_state_dict(model_state)
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except ValueError:
                # Optimizer shape changed; fall back to freshly initialized optimizer.
                pass

        self._checkpoint_metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state", "optimizer_state", "feature_dim"}
        }
        return True


if __name__ == "__main__":
    # This main guard is now for simple testing/debugging, 
    # the main training loop is in experiments/refactor_harness.py
    print("AgentV2 can be initialized, but requires a harness to run.")
