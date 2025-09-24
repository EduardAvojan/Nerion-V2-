from pathlib import Path

import torch

from nerion_digital_physicist.infrastructure.memory import ReplayStore
from nerion_digital_physicist.infrastructure.replay_sampler import sample_training_batch
from nerion_digital_physicist.infrastructure.replay_trainer import run_replay_training_step
from nerion_digital_physicist.agent.brain import CodeGraphNN


def _write_simple_module(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "def foo(x):\n    return x + 1\n",
        encoding="utf-8",
    )


def test_sample_training_batch(tmp_path: Path) -> None:
    source_path = tmp_path / "src" / "module.py"
    _write_simple_module(source_path)

    store = ReplayStore(tmp_path)
    store.append(
        task_id="task1",
        template_id="tpl",
        status="solved",
        metadata={
            "source_path": str(source_path),
            "artifacts_path": str(tmp_path),
        },
    )

    batch = sample_training_batch(tmp_path, batch_size=1)
    assert len(batch) == 1
    sample = batch[0]
    assert sample.label == 0
    assert sample.graph_data.x.shape[0] > 0


def test_run_replay_training_step(tmp_path: Path) -> None:
    source_path = tmp_path / "src" / "module.py"
    _write_simple_module(source_path)

    store = ReplayStore(tmp_path)
    store.append(
        task_id="task2",
        template_id="tpl",
        status="failed",
        metadata={"source_path": str(source_path)},
    )

    sample = sample_training_batch(tmp_path, batch_size=1)[0]
    feature_dim = sample.graph_data.x.shape[1]
    model = CodeGraphNN(num_node_features=feature_dim, hidden_channels=16, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss = run_replay_training_step(tmp_path, model, optimizer, batch_size=1)
    assert loss is not None
