"Automation harness for training the GNN on project-wide refactoring tasks."
from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

# Add project root to path to allow direct execution
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataclasses import asdict

from nerion_digital_physicist.generation.curriculum_generator import _run_test_in_sandbox
from nerion_digital_physicist.agent.policy import AgentV2
from nerion_digital_physicist.infrastructure.memory import ReplayStore
from nerion_digital_physicist.infrastructure.telemetry import TelemetryLogger
from nerion_digital_physicist.infrastructure.outcomes import log_outcome
from nerion_digital_physicist.environment.types import TestOutcome

# Type alias for experience tuples
Experience = Tuple[str, str, str, TestOutcome]


def train_agent(agent: AgentV2, replay_store: ReplayStore, graph_data: Data, node_map: Dict[int, Dict], batch_size: int):
    """Samples experiences and trains the agent."""
    if len(list(replay_store.load())) < batch_size:
        print("Not enough experiences in memory to train.")
        return

    print(f"\nSampling {batch_size} experiences and training agent...")
    experiences = replay_store.sample(batch_size)
    
    # Unpack the Experience objects into the format expected by the agent
    training_data = []
    for exp in experiences:
        training_data.append((
            exp.metadata['before_code'], 
            exp.metadata['after_code'], 
            exp.metadata['test_code'], 
            TestOutcome(**exp.metadata['outcome'])
        ))
    
    # We will pass the full batch to the agent to process
    # The agent's `learn` method will be responsible for iterating and calculating loss
    loss = agent.learn(training_data, graph_data, node_map)
    print(f"Training complete. Average Loss: {loss:.4f}")


def main():
    """Main training loop for the refactoring task."""
    # --- Config ---
    NUM_EPISODES = 10
    BATCH_SIZE = 3
    REPLAY_ROOT = Path("out/training_runs/replay")
    DB_PATH = Path("out/learning/curriculum.sqlite")

    # Clear the replay file
    if (REPLAY_ROOT / "replay.jsonl").exists():
        (REPLAY_ROOT / "replay.jsonl").unlink()

    # --- Load Lessons from DB ---
    print(f"Loading lessons from {DB_PATH}...")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM lessons")
    lessons = cur.fetchall()
    con.close()
    print(f"Loaded {len(lessons)} lessons.")

    print("\n2. Initializing environment, memory, and agent...")
    replay_store = ReplayStore(REPLAY_ROOT)
    telemetry = TelemetryLogger(REPLAY_ROOT)
    
    # This is a placeholder for the graph data. In a real scenario, this would
    # be the graph of the code that the agent is working on.
    graph_data = Data(x=torch.randn(2, 1), edge_index=torch.tensor([[0, 1],[1, 0]], dtype=torch.long), batch=torch.tensor([0, 1], dtype=torch.long))
    node_map = {}

    agent = AgentV2(input_dim=2)
    print(f"Agent initialized.")

    print(f"\n3. Gathering {NUM_EPISODES} experiences...")
    for i in range(NUM_EPISODES):
        print(f"\n--- Episode {i + 1}/{NUM_EPISODES} ---")
        lesson = random.choice(lessons)
        before_code = lesson["before_code"]
        after_code = lesson["after_code"]
        test_code = lesson["test_code"]

        before_proc = _run_test_in_sandbox(before_code, test_code)
        after_proc = _run_test_in_sandbox(after_code, test_code)

        outcome = TestOutcome(
            passed=1 if after_proc.returncode == 0 else 0,
            failed=1 if after_proc.returncode != 0 else 0,
        )

        # This is a placeholder for the surprise calculation. In a real scenario, this
        # would use the agent's model to predict the outcome and calculate the surprise.
        surprise = random.random()
        
        experience = replay_store.append(
            task_id=lesson["name"],
            template_id="lesson",
            status="solved" if outcome.was_successful else "failed",
            surprise=surprise,
            metadata={"before_code": before_code, "after_code": after_code, "test_code": test_code, "outcome": asdict(outcome)}
        )
        
        log_outcome(
            replay=replay_store,
            telemetry=telemetry,
            experience_id=experience.experience_id,
            status="solved" if outcome.was_successful else "failed",
            surprise=surprise,
        )

        print(f"Episode Complete. Outcome: {'Success' if outcome.was_successful else 'Failure'} {outcome}. Surprise: {surprise:.4f}. Memory size: {len(list(replay_store.load()))}")

    print("\nExperience gathering finished.")

    # --- Training Phase ---
    train_agent(agent, replay_store, graph_data, node_map, BATCH_SIZE)


if __name__ == "__main__":
    main()