"""Phase 2 agent: integrates curiosity-driven planning with scope-aware edits."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .brain import CodeGraphNN
from .data import create_graph_data_object
from .semantics import SemanticEmbedder, get_global_embedder
from ..environment.core import Action, EnvironmentV2


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
    ):
        self._logic_path = (
            Path(__file__).resolve().parent.parent / "environment" / "logic_v2.py"
        )
        self.embedder = embedder or get_global_embedder()
        self.env = EnvironmentV2(embedder=self.embedder)
        base_graph = _ensure_batch(
            create_graph_data_object(self._logic_path, embedder=self.embedder)
        )

        feature_dim = base_graph.x.shape[1]
        self.brain = CodeGraphNN(
            num_node_features=feature_dim,
            hidden_channels=64,
            num_classes=2,
        )
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
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
        self._action_visit_counts: dict[Action, int] = {action: 0 for action in Action}
        self._checkpoint_metadata: dict[str, Any] = {}

    def _perceive(self) -> Data:
        """Observe the current environment state as a graph."""
        return _ensure_batch(
            create_graph_data_object(self._logic_path, embedder=self.embedder)
        )

    @property
    def source_path(self) -> Path:
        """Return the path to the environment source file."""

        return self._logic_path

    def learn_from_memory(self, epochs: int = 10, *, verbose: bool = True) -> None:
        """Fine-tune the brain using accumulated experiences."""
        if not self.memory:
            return

        if verbose:
            print(f"🧠 Re-training brain on {len(self.memory)} memories...")
        self.brain.train()
        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(self.memory)
            for graph_data, label in self.memory:
                self.optimizer.zero_grad()
                out = self.brain(graph_data.x, graph_data.edge_index, graph_data.batch)
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(self.memory)
                print(f"   Epoch {epoch + 1:02d}, Avg Loss: {avg_loss:.4f}")

    def _imagine_actions(
        self, *, verbose: bool = True
    ) -> Tuple[Dict[Action, torch.Tensor], Dict[Action, Data]]:
        """Evaluate every action by predicting its post-action test outcome."""

        self.brain.eval()
        action_predictions: dict[Action, torch.Tensor] = {}
        hypothetical_graphs: dict[Action, Data] = {}

        for action in list(Action):
            hypothetical_graph = _ensure_batch(self.env.preview_action_graph(action))
            hypothetical_graphs[action] = hypothetical_graph

            with torch.no_grad():
                logits = self.brain(
                    hypothetical_graph.x,
                    hypothetical_graph.edge_index,
                    hypothetical_graph.batch,
                )
                probs = F.softmax(logits, dim=1)[0]
                action_predictions[action] = probs
                if verbose:
                    print(
                        f"  - For {action.name}: Predict Pass={probs[0]:.2f}, Fail={probs[1]:.2f}"
                    )

        return action_predictions, hypothetical_graphs

    @staticmethod
    def _entropy(probs: torch.Tensor) -> float:
        probs = probs.clamp(min=1e-12)
        entropy = -(probs * probs.log()).sum().item()
        return float(entropy)

    @staticmethod
    def _uncertainty(probs: torch.Tensor) -> float:
        return float(1.0 - abs(probs[0] - probs[1]).item())

    def _select_action(
        self,
        action_predictions: Dict[Action, torch.Tensor],
        *,
        verbose: bool = True,
    ) -> PolicyDecision:
        """Select an action using epsilon-greedy exploration with visit-aware tiebreak."""

        # Exploration: epsilon chance to pick a purely random action.
        if self.epsilon > 0 and random.random() < self.epsilon:
            chosen_action = random.choice(list(action_predictions.keys()))
            entropy = self._entropy(action_predictions[chosen_action])
            uncertainty = self._uncertainty(action_predictions[chosen_action])
            if verbose:
                print(
                    f"🤖 Agent explores via epsilon-greedy: {chosen_action.name}"
                    f" (ε={self.epsilon:.2f})"
                )
            return PolicyDecision(
                action=chosen_action,
                mode="epsilon",
                epsilon=self.epsilon,
                uncertainty=uncertainty,
                entropy=entropy,
                visit_count=self._action_visit_counts.get(chosen_action, 0),
            )

        # Curiosity-guided selection with uncertainty metric and visit-aware tie-breaker.
        best_actions: list[Action] = []
        best_score = float("-inf")
        score_cache: Dict[Action, Tuple[float, float, float]] = {}
        for action, probs in action_predictions.items():
            uncertainty = self._uncertainty(probs)
            entropy = self._entropy(probs)
            score = uncertainty + self.entropy_bonus * entropy
            score_cache[action] = (score, uncertainty, entropy)
            if score > best_score + 1e-9:
                best_score = score
                best_actions = [action]
            elif abs(score - best_score) <= 1e-9:
                best_actions.append(action)

        assert best_actions, "Action list cannot be empty."

        if len(best_actions) > 1:
            min_visit = min(self._action_visit_counts.get(action, 0) for action in best_actions)
            best_actions = [action for action in best_actions if self._action_visit_counts.get(action, 0) == min_visit]

        chosen_action = random.choice(best_actions)
        _, uncertainty, entropy = score_cache[chosen_action]
        if verbose:
            print(
                f"🤖 Agent is most curious about: {chosen_action.name}"
                f" (Score: {best_score:.2f}, visits={self._action_visit_counts.get(chosen_action, 0)})"
            )
        return PolicyDecision(
            action=chosen_action,
            mode="curiosity",
            epsilon=self.epsilon,
            uncertainty=uncertainty,
            entropy=entropy,
            visit_count=self._action_visit_counts.get(chosen_action, 0),
        )

    def run_episode(
        self,
        *,
        verbose: bool = True,
        forced_action: Action | None = None,
    ) -> EpisodeResult:
        """Execute a single episode and return the outcome summary."""

        if verbose:
            print("🤔 Imagining outcomes of possible actions...")
        action_predictions, hypothetical_graphs = self._imagine_actions(verbose=verbose)
        if forced_action is not None:
            if forced_action not in action_predictions:
                raise ValueError(f"Forced action {forced_action!r} not available")
            forced_probs = action_predictions[forced_action]
            decision = PolicyDecision(
                action=forced_action,
                mode="scheduled",
                epsilon=self.epsilon,
                uncertainty=self._uncertainty(forced_probs),
                entropy=self._entropy(forced_probs),
                visit_count=self._action_visit_counts.get(forced_action, 0),
            )
        else:
            decision = self._select_action(action_predictions, verbose=verbose)
        best_action = decision.action

        outcome_is_success = self.env.step(best_action, verbose=verbose)
        action_metadata = self.env.last_action_metadata()
        raw_tags = action_metadata.get("action_tags", ())
        if isinstance(raw_tags, str):
            action_tags = (raw_tags,)
        else:
            action_tags = tuple(str(tag) for tag in raw_tags) if raw_tags else ()
        predicted = action_predictions[best_action].detach().cpu()
        predicted_pass = float(predicted[0].item())
        predicted_fail = float(predicted[1].item())
        actual_probability = predicted_pass if outcome_is_success else predicted_fail
        surprise = max(0.0, min(1.0, 1.0 - actual_probability))

        outcome_label = torch.tensor([0 if outcome_is_success else 1], dtype=torch.long)
        experience_graph = hypothetical_graphs[best_action].clone()
        experience_graph = _ensure_batch(experience_graph)

        if verbose:
            print("🤔 Adding new experience to memory...")
        self.memory.append((experience_graph, outcome_label.clone()))

        # Update visit statistics post-selection
        self._action_visit_counts[best_action] = self._action_visit_counts.get(best_action, 0) + 1

        self.learn_from_memory(verbose=verbose)

        next_epsilon = self._update_adaptive_parameters(surprise)

        return EpisodeResult(
            action=best_action,
            predicted_pass=predicted_pass,
            predicted_fail=predicted_fail,
            outcome_is_success=outcome_is_success,
            surprise=surprise,
            memory_size=len(self.memory),
            policy_mode=decision.mode,
            policy_epsilon=decision.epsilon,
            policy_uncertainty=decision.uncertainty,
            policy_entropy=decision.entropy,
            policy_entropy_bonus=self.entropy_bonus,
            policy_visit_count=decision.visit_count,
            policy_epsilon_next=next_epsilon,
            action_tags=action_tags,
            action_metadata={k: v for k, v in action_metadata.items()},
        )

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

    def run_life_cycle(self, num_episodes: int = 3) -> None:
        """Main loop where the agent imagines, selects, acts, and learns."""
        print("🚀 Starting Agent Life Cycle...")

        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            result = self.run_episode(verbose=True)
            print(
                f"Episode result: action={result.action.name}, "
                f"success={result.outcome_is_success}, surprise={result.surprise:.2f}"
            )

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
    agent = AgentV2()
    agent.run_life_cycle()
