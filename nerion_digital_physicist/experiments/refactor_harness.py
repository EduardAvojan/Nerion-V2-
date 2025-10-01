"""
Automation harness for training the GNN on project-wide refactoring tasks.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

# Add project root to path to allow direct execution
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataclasses import asdict

from nerion_digital_physicist.agent.project_graph import ProjectParser, to_pyg_data
from nerion_digital_physicist.environment.refactor_env import RefactorEnvironment, RenameAction, TestOutcome
from nerion_digital_physicist.agent.policy import AgentV2
from nerion_digital_physicist.infrastructure.memory import ReplayStore
from nerion_digital_physicist.infrastructure.telemetry import TelemetryLogger
from nerion_digital_physicist.infrastructure.outcomes import log_outcome

from nerion_digital_physicist.infrastructure.knowledge_graph import KnowledgeGraph

# Type alias for experience tuples
Experience = Tuple[RenameAction, TestOutcome]


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

def train_agent(agent: AgentV2, replay_store: ReplayStore, graph_data: Data, node_map: Dict[int, Dict], batch_size: int):
    """Samples experiences and trains the agent."""
    if len(list(replay_store.load())) < batch_size:
        print("Not enough experiences in memory to train.")
        return

    print(f"\nSampling {batch_size} experiences and training agent...")
    experiences = replay_store.sample(batch_size)
    
    # Unpack the Experience objects into the format expected by the agent
    training_data = [(RenameAction(**exp.metadata['action']), TestOutcome(**exp.metadata['outcome'])) for exp in experiences]
    
    # We will pass the full batch to the agent to process
    # The agent's `learn` method will be responsible for iterating and calculating loss
    loss = agent.learn(training_data, graph_data, node_map)
    print(f"Training complete. Average Loss: {loss:.4f}")


def main():
    """Main training loop for the refactoring task."""
    # --- Config ---
    NUM_EPISODES = 1
    BATCH_SIZE = 3
    REPLAY_ROOT = Path("out/training_runs/replay")
    KG_PATH = Path("out/training_runs/knowledge_graph.graphml")
    CACHE_PATH = Path("project_graph_cache.pt")

    # --- Graph Loading/Building ---
    print("1. Building project graph...")
    parser = ProjectParser(".")
    parser.discover_and_parse()
    graph = parser.build_graph()
    pyg_data, node_map = to_pyg_data(graph, parser)
    print("Project graph built successfully.")

    if KG_PATH.exists():
        print(f"Loading knowledge graph from {KG_PATH}...")
        kg = KnowledgeGraph.load(KG_PATH)
    else:
        print("Creating new knowledge graph...")
        kg = KnowledgeGraph(graph)

    print("\n--- KNOWLEDGE GRAPH DEBUG ---")
    print(f"Nodes: {kg.graph.nodes(data=True)}")
    print(f"Edges: {kg.graph.edges(data=True)}")
    print("---------------------------")

    environment = RefactorEnvironment(".", knowledge_graph=kg)
    replay_store = ReplayStore(REPLAY_ROOT)
    telemetry = TelemetryLogger(REPLAY_ROOT)
    
    num_features = pyg_data.x.shape[1] + 1
    agent = AgentV2(input_dim=num_features)
    print(f"Agent initialized for {num_features}-dimensional feature space.")

    # A predefined list of refactoring tasks to attempt
    tasks = [
        # This is a known-good rename that should succeed
        RenameAction(
            file_path="nerion_digital_physicist/agent/brain.py",
            old_name="CodeGraphNN",
            new_name="GraphBrain",
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
        
        mean_prediction, uncertainty = agent.predict_with_uncertainty(pyg_data, node_map, action)
        
        total_tests = outcome.passed + outcome.failed + outcome.errored
        if total_tests == 0:
            success_score = 0.0
        else:
            success_score = outcome.passed / total_tests
            
        prediction_error = abs(mean_prediction.item() - success_score)
        surprise = prediction_error / (uncertainty.item() + 1e-6)
        
        experience = replay_store.append(
            task_id=f"refactor_{i}",
            template_id="refactoring",
            status="solved" if outcome.was_successful else "failed",
            surprise=surprise,
            metadata={"action": asdict(action), "outcome": asdict(outcome)}
        )
        
        log_outcome(
            replay=replay_store,
            telemetry=telemetry,
            experience_id=experience.experience_id,
            status="solved" if outcome.was_successful else "failed",
            surprise=surprise,
        )

        # Update knowledge graph
        action_id = f"action_{i}"
        outcome_id = f"outcome_{i}"
        file_id = action.file_path

        kg.add_node(file_id, node_type="CodeFile", path=file_id)
        kg.add_node(action_id, node_type="RefactoringAction", **asdict(action))
        kg.add_node(outcome_id, node_type="TestOutcome", **asdict(outcome))

        kg.add_edge(action_id, file_id, edge_type="MODIFIES")
        kg.add_edge(action_id, outcome_id, edge_type="HAS_OUTCOME")

        target_node_idx = find_node_index(action, node_map)
        if target_node_idx is not None:
            target_node_id = node_map[target_node_idx]["id"]
            if "::" in target_node_id:
                function_name = target_node_id.split("::")[-1]
                kg.add_node(target_node_id, node_type="Function", name=function_name)
                kg.add_edge(file_id, target_node_id, edge_type="CONTAINS")
            else:
                kg.add_node(target_node_id, node_type="CodeFile", path=target_node_id)
            kg.add_edge(action_id, target_node_id, edge_type="MODIFIES")

        print(f"Episode Complete. Outcome: {'Success' if outcome.was_successful else 'Failure'} {outcome}. Surprise: {surprise:.4f}. Memory size: {len(list(replay_store.load()))}")

    print("\nExperience gathering finished.")

    # --- Training Phase ---
    train_agent(agent, replay_store, pyg_data, node_map, BATCH_SIZE)

    # --- Save Knowledge Graph ---
    print(f"\nSaving knowledge graph to {KG_PATH}...")
    kg.save(KG_PATH)


if __name__ == "__main__":
    main()
