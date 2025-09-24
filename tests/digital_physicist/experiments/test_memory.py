from pathlib import Path

from nerion_digital_physicist.infrastructure.memory import (
    ReplayStore,
    REPLAY_FILENAME,
    DEFAULT_PRIORITY,
)


def test_append_and_load_experiences(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path)
    exp = store.append(task_id="task1", template_id="tpl", surprise=0.5)

    experiences = list(store.load())
    assert len(experiences) == 1
    assert experiences[0].experience_id == exp.experience_id
    assert experiences[0].priority > DEFAULT_PRIORITY


def test_update_experience(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path)
    exp = store.append(task_id="task2", template_id="tpl")

    updated = store.update(exp.experience_id, status="solved", surprise=0.2)
    assert updated.status == "solved"
    assert updated.surprise == 0.2

    reloaded = list(store.load())[0]
    assert reloaded.status == "solved"


def test_priority_sampling(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path)
    exp_low = store.append(task_id="low", template_id="tpl", priority=0.5)
    exp_high = store.append(task_id="high", template_id="tpl", priority=5.0)

    samples = store.sample(1, strategy="priority")
    assert samples[0].experience_id in {exp_low.experience_id, exp_high.experience_id}

    random_samples = store.sample(2, strategy="random")
    assert len(random_samples) == 2


def test_replay_file_written(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path)
    store.append(task_id="task3", template_id="tpl")
    assert (tmp_path / REPLAY_FILENAME).exists()

def test_find_by_task(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path)
    exp = store.append(task_id="task42", template_id="tpl")
    found = store.find_by_task("task42")
    assert found is not None
    assert found.experience_id == exp.experience_id
