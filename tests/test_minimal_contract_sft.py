import json

from generate_minimal_contract_sft_data import (
    MINIMAL_CONTRACT_KEYS,
    build_main_sample,
    build_sub_samples,
    minimal_mock_contract,
    parse_minimal_contract_text,
    unique_action_guidance,
)


def _step():
    return type(
        "Step",
        (),
        {
            "task": "Find chocolate",
            "observation": "This room is called the kitchen.",
            "subtask": "Search the kitchen cupboard for chocolate",
            "expert_actions": [
                "open cupboard",
                "look in cupboard",
                "look in cupboard",
                "take chocolate",
            ],
            "low_observations": ["kitchen", "open cupboard", "open cupboard again", "chocolate visible"],
            "low_dones": [False, False, False, True],
            "source_index": 0,
            "step_index": 1,
            "task_family": "find chocolate",
            "split_key": "find chocolate",
        },
    )()


def test_unique_action_guidance_preserves_first_official_actions():
    assert unique_action_guidance(["look", "0", "look", "take item", "1"], 6) == [
        "look",
        "take item",
    ]
    assert unique_action_guidance(["a", "b", "c"], 2) == ["a", "b"]


def test_minimal_main_sample_has_only_contract_interface_fields():
    step = _step()
    contract = minimal_mock_contract(step, guidance_limit=3)
    sample = build_main_sample(step, contract)
    assistant = sample["messages"][-1]["content"]
    parsed = parse_minimal_contract_text(assistant)
    payload = parsed.to_payload()

    assert set(payload) == set(MINIMAL_CONTRACT_KEYS)
    assert "rationale" not in payload
    assert "fallback_if_blocked" not in payload
    assert payload["action_guidance"] == ["open cupboard", "look in cupboard", "take chocolate"]


def test_minimal_contract_parser_rejects_old_verbose_fields():
    text = "[contract]" + json.dumps(
        {
            "subgoal": "look",
            "success_condition": "done",
            "target_objects": [],
            "action_guidance": ["look"],
            "handoff_if": "complete when done",
            "rationale": "old verbose field",
        }
    ) + "[/contract]"
    try:
        parse_minimal_contract_text(text)
    except ValueError as exc:
        assert "unexpected minimal contract keys" in str(exc)
    else:
        raise AssertionError("expected parser to reject forbidden verbose field")


def test_minimal_sub_samples_include_action_done_and_handoff():
    step = _step()
    contract = minimal_mock_contract(step, guidance_limit=3)
    samples = build_sub_samples(step, contract)

    assert samples[0]["messages"][-1]["content"] == (
        "[action]open cupboard[/action][subtask_done]false[/subtask_done]"
        "[handoff]continue[/handoff]"
    )
    assert samples[-1]["messages"][-1]["content"] == (
        "[action]take chocolate[/action][subtask_done]true[/subtask_done]"
        "[handoff]complete[/handoff]"
    )
