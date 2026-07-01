import json

import pytest

from scienceworld_mas.training import load_training_examples, transition_to_example


def test_transition_to_system1_example():
    example = transition_to_example(
        {
            "role": "system1",
            "task_description": "Boil water.",
            "observation": "You are in the kitchen.",
            "target_subgoal": "find a container with water",
        }
    )

    assert example.role == "system1"
    assert example.messages[1]["content"] == (
        "Task:\nBoil water.\n\nObservation:\nYou are in the kitchen."
    )
    assert example.messages[-1] == {
        "role": "assistant",
        "content": "find a container with water",
    }


def test_transition_to_system2_example():
    example = transition_to_example(
        {
            "role": "system2",
            "subgoal": "find a container with water",
            "observation": "A jug is on the table.",
            "target_action": "look at jug",
            "subgoal_done": True,
        }
    )

    assert example.role == "system2"
    assert example.messages[1]["content"] == (
        "Subgoal:\nfind a container with water\n\nObservation:\nA jug is on the table."
    )
    assert example.messages[-1]["content"] == (
        "[action]look at jug[/action][subgoal_done]true[/subgoal_done]"
    )


def test_transition_to_example_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown transition role"):
        transition_to_example({"role": "other"})


def test_load_training_examples_reads_role_split(tmp_path):
    role_dir = tmp_path / "system1"
    role_dir.mkdir()
    record = {
        "role": "system1",
        "task_description": "Freeze water.",
        "observation": "A freezer is nearby.",
        "target_subgoal": "put water in freezer",
    }
    (role_dir / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    examples = load_training_examples(tmp_path, role="system1", split="train")

    assert len(examples) == 1
    assert examples[0].messages[-1]["content"] == "put water in freezer"
