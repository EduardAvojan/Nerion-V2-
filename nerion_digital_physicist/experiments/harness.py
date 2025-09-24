"""Automation harness for running batches of Nerion episodes."""

from __future__ import annotations

import argparse
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch

from ..agent.policy import AgentV2, EpisodeResult, Action
from ..infrastructure.registry import ManifestRegistry, TaskManifest
from ..infrastructure.memory import ReplayStore
from ..infrastructure.outcomes import log_outcome
from ..infrastructure.telemetry import TelemetryLogger
from ..infrastructure.replay_trainer import run_replay_training_step, replay_ready
from ..infrastructure.replay_sampler import sample_training_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nerion experiment batches")
    parser.add_argument("--batches", type=int, default=1, help="Number of batches to execute")
    parser.add_argument(
        "--episodes-per-batch",
        type=int,
        default=10,
        help="Number of episodes to run per batch",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "experiment_runs"),
        help="Directory for telemetry and replay artifacts",
    )
    parser.add_argument(
        "--task-root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "generation" / "generated_tasks"),
        help="Root directory containing generated task manifests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed used for per-batch task generation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode console output",
    )
    parser.add_argument(
        "--policy-epsilon",
        type=float,
        default=0.1,
        help="Epsilon value for the curiosity policy",
    )
    parser.add_argument(
        "--replay-epochs",
        type=int,
        default=5,
        help="Replay fine-tuning epochs to run after each batch",
    )
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=4,
        help="Number of experiences per replay training step",
    )
    parser.add_argument(
        "--replay-learning-rate",
        type=float,
        default=0.005,
        help="Learning rate for replay fine-tuning",
    )
    parser.add_argument(
        "--generative-per-batch",
        type=int,
        default=0,
        help=
        "Number of episodes per batch to force the IMPLEMENT_MULTIPLY_DOCSTRING action",
    )
    parser.add_argument(
        "--entropy-bonus",
        type=float,
        default=0.0,
        help="Entropy bonus coefficient added to curiosity scoring",
    )
    parser.add_argument(
        "--adaptive-epsilon",
        action="store_true",
        help="Enable adaptive epsilon updates based on episode surprise",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.0,
        help="Minimum epsilon when adaptive updates are enabled",
    )
    parser.add_argument(
        "--epsilon-max",
        type=float,
        default=None,
        help="Maximum epsilon when adaptive updates are enabled (defaults to initial value)",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=1.0,
        help="Multiplicative decay applied when surprise is below the target",
    )
    parser.add_argument(
        "--epsilon-step",
        type=float,
        default=0.0,
        help="Additive step applied when surprise exceeds the target",
    )
    parser.add_argument(
        "--surprise-target",
        type=float,
        default=0.0,
        help="Surprise threshold used by adaptive epsilon logic",
    )
    parser.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Disable just-in-time task generation inside the experiment loop",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "checkpoints" / "nerion_brain.pth"),
        help="File path used to persist the agent brain between runs",
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="Load the checkpoint (if present) before starting the experiment",
    )
    return parser.parse_args()


@dataclass
class BatchSummary:
    batch_index: int
    episodes: int
    successes: int
    failures: int
    avg_surprise: float
    surprise_variance: float


def _status_from_result(result: EpisodeResult) -> str:
    return "solved" if result.outcome_is_success else "failed"


def _run_cli_command(
    command: Sequence[str], *, description: str, quiet: bool = False
) -> None:
    """Run a subprocess command and raise a helpful error on failure."""

    cmd_display = " ".join(command)
    if not quiet:
        print(f"[{description}] Running: {cmd_display}")

    completed = subprocess.run(  # noqa: S603 - controlled command
        command,
        capture_output=quiet,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        stdout = completed.stdout if quiet else "(see console)"
        stderr = completed.stderr if quiet else "(see console)"
        raise RuntimeError(
            f"{description} failed with exit code {completed.returncode}.\n"
            f"Command: {cmd_display}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    if not quiet and completed.stdout:
        print(completed.stdout.strip())


def _reserve_manifests(registry: ManifestRegistry, count: int) -> List[TaskManifest]:
    reserved: List[TaskManifest] = []
    candidates = registry.list_by_status("generated", "claimed")

    for manifest in candidates:
        if manifest.status != "claimed":
            manifest = registry.set_status(manifest.task_id, "claimed")
        reserved.append(manifest)
        if len(reserved) == count:
            break

    return reserved


def run_batch(
    agent: AgentV2,
    replay: ReplayStore,
    telemetry: TelemetryLogger,
    registry: ManifestRegistry,
    batch_index: int,
    episodes_per_batch: int,
    quiet: bool,
    starting_episode: int,
    *,
    replay_epochs: int,
    replay_batch_size: int,
    replay_learning_rate: float,
    generative_per_batch: int = 0,
) -> BatchSummary:
    results: List[EpisodeResult] = []
    manifests = _reserve_manifests(registry, episodes_per_batch)

    if len(manifests) < episodes_per_batch:
        raise RuntimeError(
            "Insufficient generated tasks available in registry for requested batch"
        )

    forced_indices: set[int] = set()
    if generative_per_batch > 0:
        count = min(generative_per_batch, episodes_per_batch)
        forced_indices = set(range(count))

    for offset in range(episodes_per_batch):
        global_episode = starting_episode + offset
        manifest = manifests[offset]
        task_id = manifest.task_id
        experience = replay.append(
            task_id=task_id,
            template_id=manifest.template_id,
            status="pending",
            metadata={
                "source_path": str(agent.source_path),
                "task_parameters": manifest.parameters,
                "artifacts_path": manifest.artifacts_path,
                "checksum": manifest.checksum,
                "template_id": manifest.template_id,
                "batch_index": batch_index,
                "global_episode": global_episode,
            },
        )

        forced_action = None
        if offset in forced_indices:
            forced_action = Action.IMPLEMENT_MULTIPLY_DOCSTRING

        result = agent.run_episode(verbose=not quiet, forced_action=forced_action)
        status = _status_from_result(result)
        log_outcome(
            replay,
            telemetry,
            experience_id=experience.experience_id,
            status=status,
            surprise=result.surprise,
            extra_metadata={
                "action": result.action.name,
                "predicted_pass": result.predicted_pass,
                "predicted_fail": result.predicted_fail,
                "batch_index": batch_index,
                "episode_index": offset + 1,
                "task_id": task_id,
                "template_id": manifest.template_id,
                "policy_mode": result.policy_mode,
                "policy_epsilon": result.policy_epsilon,
                "policy_uncertainty": result.policy_uncertainty,
                "policy_entropy": result.policy_entropy,
                "policy_entropy_bonus": result.policy_entropy_bonus,
                "policy_visit_count": result.policy_visit_count,
                "policy_epsilon_next": result.policy_epsilon_next,
                "action_tags": result.action_tags,
                "action_metadata": result.action_metadata,
            },
        )

        manifest_status = "solved" if result.outcome_is_success else "archived"
        registry.set_status(task_id, manifest_status)

        telemetry.log(
            "episode_completed",
            {
                "batch_index": batch_index,
                "episode_index": offset + 1,
                "status": status,
                "surprise": result.surprise,
                "predicted_pass": result.predicted_pass,
                "predicted_fail": result.predicted_fail,
                "memory_size": result.memory_size,
                "task_id": task_id,
                "template_id": manifest.template_id,
                "policy_mode": result.policy_mode,
                "policy_epsilon": result.policy_epsilon,
                "policy_uncertainty": result.policy_uncertainty,
                "policy_entropy": result.policy_entropy,
                "policy_entropy_bonus": result.policy_entropy_bonus,
                "policy_visit_count": result.policy_visit_count,
                "policy_epsilon_next": result.policy_epsilon_next,
                "action_tags": result.action_tags,
                "action_metadata": result.action_metadata,
            },
        )

        results.append(result)

    successes = sum(1 for r in results if r.outcome_is_success)
    failures = len(results) - successes
    surprises = [r.surprise for r in results]
    avg_surprise = statistics.fmean(surprises) if surprises else 0.0
    surprise_variance = statistics.pvariance(surprises) if len(surprises) > 1 else 0.0

    telemetry.log(
        "batch_completed",
        {
            "batch_index": batch_index,
            "episodes": len(results),
            "successes": successes,
            "failures": failures,
            "avg_surprise": avg_surprise,
            "surprise_variance": surprise_variance,
        },
    )

    if not quiet:
        print(
            f"Batch {batch_index}: {len(results)} episodes, "
            f"successes={successes}, failures={failures}, avg_surprise={avg_surprise:.3f}"
        )

    summary = BatchSummary(
        batch_index=batch_index,
        episodes=len(results),
        successes=successes,
        failures=failures,
        avg_surprise=avg_surprise,
        surprise_variance=surprise_variance,
    )

    _run_replay_finetuning(
        agent,
        replay,
        telemetry,
        batch_index=batch_index,
        epochs=replay_epochs,
        batch_size=replay_batch_size,
        learning_rate=replay_learning_rate,
        quiet=quiet,
    )

    return summary


def _run_replay_finetuning(
    agent: AgentV2,
    replay: ReplayStore,
    telemetry: TelemetryLogger,
    *,
    batch_index: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    quiet: bool,
) -> None:
    if epochs <= 0 or batch_size <= 0:
        return

    if not replay_ready(replay.root):
        if not quiet:
            print("No replay data available for fine-tuning; skipping.")
        return

    initial_batch = sample_training_batch(replay.root, batch_size=1)
    if not initial_batch:
        if not quiet:
            print("Unable to obtain replay samples; skipping fine-tuning.")
        return

    feature_dim = initial_batch[0].graph_data.x.shape[1]
    if feature_dim != agent.brain.conv1.in_channels:
        # Initialize a new brain to match feature dimensionality, then copy weights where possible
        agent.brain = agent.brain.__class__(
            num_node_features=feature_dim,
            hidden_channels=agent.brain.conv1.out_channels,
            num_classes=2,
        )
        agent.optimizer = torch.optim.Adam(agent.brain.parameters(), lr=learning_rate)

    agent.brain.train()
    agent.optimizer = torch.optim.Adam(agent.brain.parameters(), lr=learning_rate)

    losses: List[float] = []
    for epoch in range(1, epochs + 1):
        loss = run_replay_training_step(
            replay_root=replay.root,
            model=agent.brain,
            optimizer=agent.optimizer,
            batch_size=batch_size,
        )
        if loss is None:
            break
        losses.append(loss)
        if not quiet:
            print(f"Replay epoch {epoch:02d}, loss={loss:.4f}")

    if not losses:
        return

    telemetry.log(
        "replay_finetune",
        {
            "batch_index": batch_index,
            "epochs": len(losses),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_min": min(losses),
            "loss_max": max(losses),
            "loss_avg": sum(losses) / len(losses),
        },
    )


def run_experiment(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    task_root = Path(args.task_root)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    telemetry = TelemetryLogger(output_root)
    replay = ReplayStore(output_root)
    registry = ManifestRegistry(task_root)
    adaptive_flag = (
        args.adaptive_epsilon
        or args.epsilon_step != 0.0
        or args.epsilon_decay != 1.0
        or args.surprise_target != 0.0
    )

    agent = AgentV2(
        epsilon=args.policy_epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        epsilon_decay=args.epsilon_decay,
        epsilon_step=args.epsilon_step,
        adaptive_surprise_target=args.surprise_target,
        adaptive_epsilon=adaptive_flag,
        entropy_bonus=args.entropy_bonus,
    )

    if checkpoint_path and args.load_checkpoint:
        loaded = agent.load_brain(checkpoint_path)
        if not args.quiet:
            if loaded:
                print(f"Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"Checkpoint not found at {checkpoint_path}; starting fresh")

    summaries: List[BatchSummary] = []
    global_episode = 1
    auto_generate = not getattr(args, "no_auto_generate", False)
    service_cli = Path(__file__).resolve().parent.parent / "generation" / "service.py"

    for batch_index in range(1, args.batches + 1):
        if auto_generate:
            batch_seed = args.seed + batch_index
            generation_cmd = [
                sys.executable,
                str(service_cli),
                str(args.episodes_per_batch),
                "--output",
                str(task_root),
                "--seed",
                str(batch_seed),
                "--curriculum",
            ]
            _run_cli_command(
                generation_cmd,
                description=f"task generation for batch {batch_index}",
                quiet=args.quiet,
            )

        summary = run_batch(
            agent,
            replay,
            telemetry,
            registry,
            batch_index,
            args.episodes_per_batch,
            args.quiet,
            global_episode,
            replay_epochs=args.replay_epochs,
            replay_batch_size=args.replay_batch_size,
            replay_learning_rate=args.replay_learning_rate,
            generative_per_batch=getattr(args, "generative_per_batch", 0),
        )
        summaries.append(summary)
        global_episode += args.episodes_per_batch

        if checkpoint_path:
            agent.save_brain(checkpoint_path)
            if not args.quiet:
                print(f"Saved checkpoint to {checkpoint_path}")

    total_episodes = sum(s.episodes for s in summaries)
    total_successes = sum(s.successes for s in summaries)
    total_failures = sum(s.failures for s in summaries)
    avg_surprise = (
        statistics.fmean([s.avg_surprise for s in summaries])
        if summaries
        else 0.0
    )

    if not args.quiet:
        print("\nExperiment complete")
        print(
            f"Episodes={total_episodes}, successes={total_successes}, "
            f"failures={total_failures}, mean(batch_avg_surprise)={avg_surprise:.3f}"
        )

    telemetry.log(
        "experiment_completed",
        {
            "batches": args.batches,
            "episodes_per_batch": args.episodes_per_batch,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "mean_batch_avg_surprise": avg_surprise,
        },
    )


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
