"""
Automation harness for training the GNN on project-wide refactoring tasks.
"""
from __future__ import annotations

import random
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

# Add project root to path to allow direct execution
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from nerion_digital_physicist.agent.project_graph import ProjectParser, to_pyg_data
from nerion_digital_physicist.environment.refactor_env import RefactorEnvironment, RenameAction, TestOutcome
from nerion_digital_physicist.agent.policy import AgentV2

# Type alias for experience tuples
Experience = Tuple[RenameAction, TestOutcome]

class ReplayMemory:
    """A simple FIFO buffer for storing experiences."""
    def __init__(self, capacity: int):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Save an experience."""
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

def find_node_index(action: RenameAction, node_map: Dict[int, Dict]) -> Optional[int]:
    """Find the graph node index corresponding to a rename action."""
    for idx, node_info in node_map.items():
        # The node_id in the map is structured like 'path/to/file.py::ClassName::method_name'
        node_id_parts = node_info.get("id", "").split("::")
        file_path = node_id_parts[0]
        
        # Match file path and the name of the function/class being renamed
        if file_path == action.file_path and node_info.get("name") == action.old_name:
            return idx
    return None

def train_agent(agent: AgentV2, memory: ReplayMemory, graph_data: Data, node_map: Dict[int, Dict], batch_size: int):
    """Samples experiences and trains the agent."""
    if len(memory) < batch_size:
        print("Not enough experiences in memory to train.")
        return

    print(f"\nSampling {batch_size} experiences and training agent...")
    experiences = memory.sample(batch_size)
    
    # We will pass the full batch to the agent to process
    # The agent's `learn` method will be responsible for iterating and calculating loss
    loss = agent.learn(experiences, graph_data, node_map)
    print(f"Training complete. Average Loss: {loss:.4f}")


def main():
    """Main training loop for the refactoring task."""
    # --- Config ---
    NUM_EPISODES = 10
    BATCH_SIZE = 3
    MEMORY_CAPACITY = 1000
    CACHE_PATH = Path("project_graph_cache.pt")

    # --- Graph Loading/Building ---
    if CACHE_PATH.exists():
        print(f"1. Loading cached project graph from {CACHE_PATH}...")
        # weights_only=False is required to load complex objects like our graph data.
        # This is safe because we are loading a file we created ourselves.
        cached_data = torch.load(CACHE_PATH, weights_only=False)
        pyg_data = cached_data["pyg_data"]
        node_map = cached_data["node_map"]
        print(f"Cached graph loaded successfully: {pyg_data}")
    else:
        print("1. Building project graph (this will be slow on the first run)...")
        parser = ProjectParser(".")
        parser.discover_and_parse()
        graph = parser.build_graph()
        pyg_data, node_map = to_pyg_data(graph, parser)
        print("Project graph built successfully. Caching for future runs...")
        torch.save({"pyg_data": pyg_data, "node_map": node_map}, CACHE_PATH)
        print(f"Graph cached to {CACHE_PATH}")

    print("\n2. Initializing environment, memory, and agent...")
    environment = RefactorEnvironment(".")
    memory = ReplayMemory(MEMORY_CAPACITY)
    
    # The agent's brain needs an extra input feature to encode the action
    num_features = pyg_data.x.shape[1] + 1
    agent = AgentV2(input_dim=num_features)
    print(f"Agent initialized for {num_features}-dimensional feature space.")

    # A predefined list of refactoring tasks to attempt
    tasks = [
        # This is a known-good rename that should succeed
        RenameAction(
            file_path="tmp_impact_target.py",
            old_name="foo",
            new_name="bar",
        ),
        RenameAction(
            file_path="selfcoder/planner/planner.py",
            old_name="plan_edits_from_nl",
            new_name="plan_edits_from_natural_language",
        ),
        RenameAction(
            file_path="selfcoder/vcs/git_ops.py",
            old_name="should_skip",
            new_name="should_ignore_path",
        ),
        # This one is expected to fail tests, providing a good negative example
        RenameAction(
            file_path="nerion_digital_physicist/agent/brain.py",
            old_name="CodeGraphNN",
            new_name="GraphBrain",
        ),
    ]

    print(f"\n3. Gathering {NUM_EPISODES} experiences...")
    for i in range(NUM_EPISODES):
        print(f"\n--- Episode {i + 1}/{NUM_EPISODES} ---")
        action = random.choice(tasks)
        print(f"Selected Action: Rename '{action.old_name}' to '{action.new_name}' in {action.file_path}")

        outcome = environment.step(action)
        memory.push((action, outcome))

        print(f"Episode Complete. Outcome: {'Success' if outcome.was_successful else 'Failure'} {outcome}. Memory size: {len(memory)}")

    print("\nExperience gathering finished.")

    # --- Training Phase ---
    train_agent(agent, memory, pyg_data, node_map, BATCH_SIZE)


if __name__ == "__main__":
    main()
