"Automation harness for training the GNN on project-wide refactoring tasks."
from __future__ import annotations

import random
from pathlib import Path
from nerion_digital_physicist.db.curriculum_store import CurriculumStore
from typing import Dict, List, Optional, Tuple
import csv

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
from nerion_digital_physicist.environment.bug_fixing_env import BugFixingEnvironment
from nerion_digital_physicist.environment.feature_implementation_env import FeatureImplementationEnvironment
from nerion_digital_physicist.environment.performance_optimization_env import PerformanceOptimizationEnvironment
from nerion_digital_physicist.environment.code_explanation_env import CodeExplanationEnvironment

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
        if exp.template_id == "lesson":
            training_data.append((
                exp.metadata['before_code'], 
                exp.metadata['after_code'], 
                exp.metadata['test_code'], 
                TestOutcome(**exp.metadata['outcome'])
            ))
        elif exp.template_id == "bug_fix":
            training_data.append((
                exp.metadata['buggy_code'], 
                exp.metadata['fixed_code'], 
                exp.metadata['test_code'], 
                TestOutcome(**exp.metadata['fixed_outcome'])
            ))
        elif exp.template_id == "feature_implementation":
            training_data.append((
                exp.metadata['initial_code'], 
                exp.metadata['final_code'], 
                exp.metadata['test_code'], 
                TestOutcome(**exp.metadata['final_outcome'])
            ))
        elif exp.template_id == "performance_optimization":
            training_data.append((
                exp.metadata['inefficient_code'], 
                exp.metadata['optimized_code'], 
                exp.metadata['test_code'], 
                exp.metadata['optimized_time'] < exp.metadata['inefficient_time']
            ))
        elif exp.template_id == "code_comprehension":
            training_data.append((
                exp.metadata['code_snippet'], 
                exp.metadata['explanation'], 
                None, 
                exp.metadata['similarity_score']
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
    with CurriculumStore(DB_PATH) as store:
        lessons = store.get_lessons_batch(limit=1000)  # Load in batches to avoid memory issues
    print(f"Loaded {len(lessons)} lessons.")

    print("\n2. Initializing environment, memory, and agent...")
    bug_fixing_env = BugFixingEnvironment(".")
    feature_implementation_env = FeatureImplementationEnvironment(".")
    performance_optimization_env = PerformanceOptimizationEnvironment(".")
    code_explanation_env = CodeExplanationEnvironment()
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
        if lesson["type"] == "refactoring":
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
        elif lesson["type"] == "bug_fixing":
            buggy_code = lesson["buggy_code"]
            test_code = lesson["test_code"]
            fixed_code = lesson["fixed_code"]

            buggy_outcome, fixed_outcome = bug_fixing_env.step(buggy_code, test_code, fixed_code)

            # This is a placeholder for the surprise calculation. In a real scenario, this
            # would use the agent's model to predict the outcome and calculate the surprise.
            surprise = random.random()
            
            experience = replay_store.append(
                task_id=lesson["name"],
                template_id="bug_fix",
                status="solved" if fixed_outcome.was_successful else "failed",
                surprise=surprise,
                metadata={"buggy_code": buggy_code, "test_code": test_code, "fixed_code": fixed_code, "buggy_outcome": asdict(buggy_outcome), "fixed_outcome": asdict(fixed_outcome)}
            )
            
            log_outcome(
                replay=replay_store,
                telemetry=telemetry,
                experience_id=experience.experience_id,
                status="solved" if fixed_outcome.was_successful else "failed",
                surprise=surprise,
            )

            print(f"Episode Complete. Buggy Outcome: {buggy_outcome}. Fixed Outcome: {fixed_outcome}. Surprise: {surprise:.4f}. Memory size: {len(list(replay_store.load()))}")
        elif lesson["type"] == "feature_implementation":
            initial_code = lesson["initial_code"]
            test_code = lesson["test_code"]
            final_code = lesson["final_code"]

            initial_outcome, final_outcome = feature_implementation_env.step(initial_code, test_code, final_code)

            # This is a placeholder for the surprise calculation. In a real scenario, this
            # would use the agent's model to predict the outcome and calculate the surprise.
            surprise = random.random()
            
            experience = replay_store.append(
                task_id=lesson["name"],
                template_id="feature_implementation",
                status="solved" if final_outcome.was_successful else "failed",
                surprise=surprise,
                metadata={"initial_code": initial_code, "test_code": test_code, "final_code": final_code, "initial_outcome": asdict(initial_outcome), "final_outcome": asdict(final_outcome)}
            )
            
            log_outcome(
                replay=replay_store,
                telemetry=telemetry,
                experience_id=experience.experience_id,
                status="solved" if final_outcome.was_successful else "failed",
                surprise=surprise,
            )

            print(f"Episode Complete. Initial Outcome: {initial_outcome}. Final Outcome: {final_outcome}. Surprise: {surprise:.4f}. Memory size: {len(list(replay_store.load()))}")
        elif lesson["type"] == "performance_optimization":
            inefficient_code = lesson["inefficient_code"]
            test_code = lesson["test_code"]
            optimized_code = lesson["optimized_code"]

            inefficient_time, optimized_time = performance_optimization_env.step(inefficient_code, test_code, optimized_code)

            # This is a placeholder for the surprise calculation. In a real scenario, this
            # would use the agent's model to predict the outcome and calculate the surprise.
            surprise = random.random()
            
            experience = replay_store.append(
                task_id=lesson["name"],
                template_id="performance_optimization",
                status="solved" if optimized_time < inefficient_time else "failed",
                surprise=surprise,
                metadata={"inefficient_code": inefficient_code, "test_code": test_code, "optimized_code": optimized_code, "inefficient_time": inefficient_time, "optimized_time": optimized_time}
            )
            
            log_outcome(
                replay=replay_store,
                telemetry=telemetry,
                experience_id=experience.experience_id,
                status="solved" if optimized_time < inefficient_time else "failed",
                surprise=surprise,
            )

            print(f"Episode Complete. Inefficient Time: {inefficient_time:.4f}s. Optimized Time: {optimized_time:.4f}s. Surprise: {surprise:.4f}. Memory size: {len(list(replay_store.load()))}")
        elif lesson["type"] == "code_comprehension":
            code_snippet = lesson["code_snippet"]
            explanation = lesson["explanation"]

            similarity_score = code_explanation_env.step(code_snippet, explanation)

            # This is a placeholder for the surprise calculation. In a real scenario, this
            # would use the agent's model to predict the outcome and calculate the surprise.
            surprise = random.random()
            
            experience = replay_store.append(
                task_id=lesson["name"],
                template_id="code_comprehension",
                status="solved" if similarity_score > 0.7 else "failed",
                surprise=surprise,
                metadata={"code_snippet": code_snippet, "explanation": explanation, "similarity_score": similarity_score}
            )
            
            log_outcome(
                replay=replay_store,
                telemetry=telemetry,
                experience_id=experience.experience_id,
                status="solved" if similarity_score > 0.7 else "failed",
                surprise=surprise,
            )

            print(f"Episode Complete. Similarity Score: {similarity_score:.4f}. Surprise: {surprise:.4f}. Memory size: {len(list(replay_store.load()))}")

    print("\nExperience gathering finished.")

    # --- Training Phase ---
    train_agent(agent, replay_store, graph_data, node_map, BATCH_SIZE)


if __name__ == "__main__":
    main()