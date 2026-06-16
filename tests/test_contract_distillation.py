import json

from contract_schema import build_mock_contract, parse_contract_text
from generate_contract_sft_data import (
    align_contract_to_expert_actions,
    build_main_sample,
    build_sub_samples,
    extract_first_json_object,
    iter_expert_steps,
)


def test_contract_round_trip():
    contract = build_mock_contract(
        task="Boil water",
        subtask="Navigate to kitchen",
        expert_actions=["open door to kitchen", "go to kitchen"],
        observation="This room is called the hallway.",
    )
    parsed = parse_contract_text(contract.to_tagged_json())
    assert parsed.subgoal == "Navigate to kitchen"
    assert "open door to kitchen" in parsed.action_guidance
    assert parsed.success_condition


def test_contract_parser_removes_kimi_separator_noise():
    parsed = parse_contract_text(
        '{"goal":"g","subgoal":"go to kitchen бк move there","rationale":"r",'
        '"success_condition":"done","action_guidance":["open door бк access room"]}'
    )
    assert parsed.subgoal == "go to kitchen - move there"
    assert parsed.action_guidance == ["open door - access room"]


def test_iter_expert_steps_aligns_high_and_low(tmp_path):
    data_dir = tmp_path / "ScienceWorld"
    data_dir.mkdir()
    (data_dir / "expert_high-data.json").write_text(
        json.dumps(
            {
                "task_description": ["Task Description:\nBoil water"],
                "obs": [["hallway", "kitchen"]],
                "subtask": [["Navigate to kitchen"]],
                "reward": [[0]],
                "score": [[0]],
                "done": [[False]],
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "expert_low-data.json").write_text(
        json.dumps(
            {
                "subtask": ["Subtask: Navigate to kitchen"],
                "obs": [["hallway", "door open", "kitchen"]],
                "action": [["open door to kitchen; False", "go to kitchen; True"]],
                "reward": [[0, 1]],
                "score": [[0, 1]],
                "done": [[False, True]],
            }
        ),
        encoding="utf-8",
    )
    steps = iter_expert_steps(data_dir)
    assert len(steps) == 1
    assert steps[0].subtask == "Navigate to kitchen"
    assert steps[0].expert_actions == ["open door to kitchen", "go to kitchen"]


def test_contract_samples_keep_expert_actions():
    step = type(
        "Step",
        (),
        {
            "task": "Boil water",
            "observation": "hallway",
            "subtask": "Navigate to kitchen",
            "expert_actions": ["open door to kitchen", "go to kitchen"],
            "low_observations": ["hallway", "door open"],
            "low_dones": [False, True],
            "source_index": 0,
            "step_index": 0,
            "task_family": "boil water",
        },
    )()
    contract = build_mock_contract(
        task=step.task,
        subtask=step.subtask,
        expert_actions=step.expert_actions,
        observation=step.observation,
    )
    main = build_main_sample(step, contract)
    sub = build_sub_samples(step, contract)
    assert main["messages"][-1]["content"].startswith("[contract]")
    assert sub[0]["messages"][-1]["content"] == (
        "[action]open door to kitchen[/action][subtask_done]false[/subtask_done]"
    )
    assert sub[1]["messages"][-1]["content"] == (
        "[action]go to kitchen[/action][subtask_done]true[/subtask_done]"
    )


def test_extract_first_json_object_from_agent_output():
    text = '• ```json\n{"goal":"g","action_guidance":["look"]}\n```\n• resume hint'
    assert json.loads(extract_first_json_object(text)) == {
        "goal": "g",
        "action_guidance": ["look"],
    }


def test_align_contract_to_expert_actions_forces_official_prefix():
    contract = build_mock_contract(
        task="Boil water",
        subtask="Heat water",
        expert_actions=["move the pot to stove", "turn on stove"],
        observation="kitchen",
    )
    noisy = type(contract)(
        goal=contract.goal,
        subgoal=contract.subgoal,
        rationale=contract.rationale,
        target_objects=contract.target_objects,
        location_hint=contract.location_hint,
        required_tools=contract.required_tools,
        success_condition=contract.success_condition,
        action_guidance=["move the pot to stove carefully", "turn on the stove", "watch temperature"],
        fallback_if_blocked=contract.fallback_if_blocked,
    )
    aligned = align_contract_to_expert_actions(
        noisy,
        ["move metal pot to stove", "activate stove"],
    )
    assert aligned.action_guidance[:2] == ["move metal pot to stove", "activate stove"]
    assert "watch temperature" in aligned.action_guidance
