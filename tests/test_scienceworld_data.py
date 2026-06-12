import json

import pytest
import torch

from scienceworld_data import (
    assign_split,
    load_high_trajectories,
    load_low_trajectories,
    parse_action_done,
    strip_embedded_instruction,
)
from provenance import portable_reference
from sft_trainer import ensure_torch_set_submodule


def test_parse_action_done():
    assert parse_action_done("open door to kitchen; False") == ("open door to kitchen", False)
    assert parse_action_done("go to kitchen; True") == ("go to kitchen", True)
    with pytest.raises(ValueError):
        parse_action_done("look around")


def test_split_is_stable():
    assert assign_split("boil water", seed=123) == assign_split("boil water", seed=123)


def test_strip_embedded_instruction():
    assert strip_embedded_instruction("planner text\n Task Description:\nBoil water.", "Task Description:") == "Boil water."


def test_load_minimal_trajectories(tmp_path):
    high_path = tmp_path / "high.json"
    high_path.write_text(
        json.dumps(
            {
                "task_description": ["Task Description: Boil water"],
                "obs": [["state", "terminal state"]],
                "subtask": [["go kitchen"]],
                "reward": [[1]],
                "score": [[1]],
                "done": [[True]],
            }
        ),
        encoding="utf-8",
    )
    low_path = tmp_path / "low.json"
    low_path.write_text(
        json.dumps(
            {
                "subtask": ["Subtask: go kitchen"],
                "obs": [["hallway", "kitchen"]],
                "action": [["go kitchen; True"]],
                "reward": [[1]],
                "score": [[1]],
                "done": [[True]],
            }
        ),
        encoding="utf-8",
    )

    high = load_high_trajectories(high_path)[0]
    assert high.subtasks == ["go kitchen"]
    assert high.observations[-1] == "terminal state"
    assert load_low_trajectories(low_path)[0].actions == ["go kitchen; True"]


def test_set_submodule_compatibility():
    ensure_torch_set_submodule()
    root = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(2, 2)))
    replacement = torch.nn.Linear(2, 1)
    root.set_submodule("0.0", replacement)
    assert root[0][0] is replacement


def test_portable_reference_hides_external_parent(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    reference = portable_reference(str(model), root=tmp_path / "different-root")
    assert reference["type"] == "external_local_path"
    assert reference["value"] == "model"
    assert str(tmp_path) not in str(reference)
