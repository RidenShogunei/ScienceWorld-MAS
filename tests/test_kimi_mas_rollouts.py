from collect_kimi_mas_rollouts import (
    format_valid_actions,
    parse_contract_response,
    parse_sub_response,
    select_candidate_actions,
    snap_action_to_valid,
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


def test_parse_contract_response_from_escaped_tagged_json():
    contract = parse_contract_response(
        '[contract]{\\"goal\\":\\"g\\",\\"subgoal\\":\\"s\\",\\"rationale\\":\\"r\\",'
        '\\"success_condition\\":\\"done\\",\\"action_guidance\\":[\\"look around\\"]}[/contract]'
    )
    assert contract is not None
    assert contract.subgoal == "s"


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


def test_format_valid_actions_prioritizes_task_actions_over_graph_actions():
    text = format_valid_actions(
        [
            "connect air to cupboard",
            "open cupboard",
            "go to kitchen",
            "disconnect door",
            "pick up thermometer",
        ],
        max_actions=3,
        max_chars=200,
        context="Need to open the cupboard and pick up the thermometer.",
    )
    lines = text.splitlines()[:3]
    assert "- open cupboard" in lines
    assert "- pick up thermometer" in lines
    assert not any("connect air" in line for line in lines)


def test_select_candidate_actions_filters_graph_actions_when_possible():
    actions = ["connect door to hallway", "disconnect air", "look around", "open cupboard"]
    selected = select_candidate_actions(
        actions,
        max_actions=10,
        include_graph_actions=False,
        context="open cupboard",
    )
    assert selected == ["look around", "open cupboard"]


def test_snap_action_to_valid_maps_near_miss():
    assert (
        snap_action_to_valid(
            "open large cupboard",
            ["open cupboard", "look around", "go to hallway"],
            threshold=0.7,
        )
        == "open cupboard"
    )
    assert snap_action_to_valid("totally unrelated", ["look around"], threshold=0.9) == "totally unrelated"


def test_snap_action_to_valid_does_not_map_task_action_to_graph_action():
    assert (
        snap_action_to_valid(
            "open door to hallway",
            ["connect door to hallway", "look around"],
            threshold=0.7,
        )
        == "open door to hallway"
    )
