from collect_kimi_mas_rollouts import (
    format_valid_actions,
    parse_contract_response,
    parse_sub_response,
)


def test_parse_contract_response_from_tagged_json():
    contract = parse_contract_response(
        '[contract]{"goal":"g","subgoal":"s","rationale":"r",'
        '"success_condition":"done","action_guidance":["look around"]}[/contract]'
    )
    assert contract is not None
    assert contract.subgoal == "s"
    assert contract.action_guidance == ["look around"]


def test_parse_contract_response_from_wrapped_json():
    contract = parse_contract_response(
        '• ```json\n{"goal":"g","subgoal":"s","rationale":"r",'
        '"success_condition":"done","action_guidance":["look around"]}\n```'
    )
    assert contract is not None
    assert contract.goal == "g"


def test_parse_sub_response():
    action, done, valid = parse_sub_response(
        "[action]open door to kitchen[/action][subtask_done]true[/subtask_done]"
    )
    assert action == "open door to kitchen"
    assert done is True
    assert valid is True


def test_format_valid_actions_limits_count_and_chars():
    text = format_valid_actions(["z", "a", "long action name"], max_actions=2, max_chars=40)
    assert "- a" in text
    assert "- long action name" in text
    assert "truncated" in text
