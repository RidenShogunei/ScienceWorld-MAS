import json

import pytest

from scienceworld_mas.data import (
    build_transition_datasets,
    load_high_trajectories,
    load_low_trajectories,
)
from scienceworld_mas.data.transitions import parse_action_done, write_transition_datasets


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def high_payload():
    return {
        "task_description": ["Prompt\nTask Description: Boil water."],
        "obs": [["obs 0", "obs 1", "obs final"]],
        "subtask": [["find water", "heat water", "unlabeled final"]],
        "reward": [[1, 99]],
        "score": [[1, 100]],
        "done": [[False, True]],
    }


def low_payload():
    return {
        "subtask": ["Prompt\nSubtask: heat water"],
        "obs": [["obs 0", "obs 1"]],
        "action": [["pick up water;false", "activate stove;true"]],
        "reward": [[1, 99]],
        "score": [[1, 100]],
        "done": [[False, True]],
    }


def test_load_high_trajectories_drops_single_unlabeled_final_subtask(tmp_path):
    path = tmp_path / "expert_high-data.json"
    write_json(path, high_payload())
    high = load_high_trajectories(path)
    assert len(high) == 1
    assert high[0].subtasks == ("find water", "heat water")
    assert high[0].dropped_unlabeled_subtasks == 1


def test_load_low_trajectories_and_parse_action_done(tmp_path):
    path = tmp_path / "expert_low-data.json"
    write_json(path, low_payload())
    low = load_low_trajectories(path)
    assert len(low) == 1
    assert parse_action_done(low[0].actions[1]) == ("activate stove", True)
    with pytest.raises(ValueError):
        parse_action_done("activate stove")


def test_build_transition_datasets_keeps_roles_separate(tmp_path):
    high_path = tmp_path / "expert_high-data.json"
    low_path = tmp_path / "expert_low-data.json"
    write_json(high_path, high_payload())
    write_json(low_path, low_payload())
    datasets = build_transition_datasets(
        load_high_trajectories(high_path),
        load_low_trajectories(low_path),
        train_ratio=0.99,
        val_ratio=0.0,
    )
    assert datasets.manifest.system1_counts["train"] == 2
    assert datasets.manifest.system2_counts["train"] == 2
    system1 = datasets.system1["train"][0]
    system2 = datasets.system2["train"][1]
    assert system1.task_description == "Boil water."
    assert system1.target_subgoal == "find water"
    assert system2.subgoal == "heat water"
    assert system2.target_action == "activate stove"
    assert system2.subgoal_done
    assert system2.episode_done


def test_write_transition_datasets(tmp_path):
    high_path = tmp_path / "expert_high-data.json"
    low_path = tmp_path / "expert_low-data.json"
    write_json(high_path, high_payload())
    write_json(low_path, low_payload())
    datasets = build_transition_datasets(
        load_high_trajectories(high_path),
        load_low_trajectories(low_path),
        train_ratio=0.99,
        val_ratio=0.0,
    )
    output = write_transition_datasets(datasets, tmp_path / "processed")
    assert (output / "manifest.json").exists()
    assert (output / "system1" / "train.jsonl").exists()
    assert (output / "system2" / "train.jsonl").exists()
    first = json.loads((output / "system1" / "train.jsonl").read_text().splitlines()[0])
    assert first["role"] == "system1"
