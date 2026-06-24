from argparse import Namespace

from generate_expert_subtask_contract_sft import (
    causal_action_guidance,
    contract_for_step,
    explicit_target_conflict,
    main_sample,
    mock_enrichment,
    recent_history,
    sub_samples,
)


def _step():
    return type(
        "Step",
        (),
        {
            "source_index": 7,
            "step_index": 2,
            "task": "Boil water",
            "observation": "Current observation: You are in the hallway.",
            "subtask": "Navigate to kitchen",
            "expert_actions": ["open door to kitchen", "go to kitchen"],
            "low_observations": ["The kitchen door is closed.", "The door is open."],
            "low_dones": [False, True],
            "task_family": "boil water",
        },
    )()


def _args():
    return Namespace(guidance_limit=6, history_limit=6, include_history=False)


def test_contract_keeps_expert_subtask_immutable():
    step = _step()
    enrichment = {
        "success_condition": "The agent is in the kitchen.",
        "target_objects": ["kitchen", "door to kitchen"],
        "subgoal": "This forbidden replacement must be ignored.",
    }
    contract = contract_for_step(step, enrichment, _args())
    assert contract.subgoal == "Navigate to kitchen"
    assert contract.action_guidance == [
        "Use available navigation actions to complete: Navigate to kitchen"
    ]


def test_main_sample_records_causal_expert_subgoal():
    step = _step()
    contract = contract_for_step(step, mock_enrichment(step), _args())
    sample = main_sample(step, contract)
    assert sample["expert_subgoal"] == "Navigate to kitchen"
    assert sample["causal_subgoal"] is True
    assert '"subgoal":"Navigate to kitchen"' in sample["messages"][-1]["content"]


def test_sub_samples_use_simple_contract_observation_prompt_by_default():
    step = _step()
    args = _args()
    contract = contract_for_step(step, mock_enrichment(step), args)
    rows = sub_samples(step, contract, args)
    assert rows[0]["valid_actions_available"] is False
    assert rows[0]["history_available"] is False
    assert "Recent execution history:" not in rows[0]["messages"][-2]["content"]
    assert rows[0]["messages"][-2]["content"].startswith("Contract:\n")
    assert "\n\nObservation:\n" in rows[0]["messages"][-2]["content"]
    assert rows[1]["messages"][-1]["content"].endswith(
        "[subtask_done]true[/subtask_done][handoff]complete[/handoff]"
    )


def test_sub_samples_can_include_history_for_ablation():
    step = _step()
    args = _args()
    args.include_history = True
    contract = contract_for_step(step, mock_enrichment(step), args)
    rows = sub_samples(step, contract, args)
    assert rows[1]["history_available"] is True
    assert "Recent execution history:" in rows[1]["messages"][-2]["content"]
    assert "open door to kitchen" in rows[1]["messages"][-2]["content"]


def test_recent_history_is_bounded():
    step = _step()
    assert recent_history(step, 1, 1) == [
        {"action": "open door to kitchen", "subtask_done": False}
    ]


def test_causal_guidance_uses_only_subtask():
    assert causal_action_guidance("Find the thermometer", 6) == [
        "Inspect relevant locations and containers to complete: Find the thermometer"
    ]


def test_explicit_box_target_conflict_is_detected():
    step = _step()
    step.subtask = "Move substance B to the green box"
    step.expert_actions = ["move substance B to blue box"]
    assert explicit_target_conflict(step) is not None
