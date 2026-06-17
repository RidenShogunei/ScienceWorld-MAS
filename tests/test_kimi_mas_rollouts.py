from collect_kimi_mas_rollouts import (
    action_from_id,
    action_rank,
    candidate_action_map,
    format_valid_actions,
    is_safe_focus_action,
    no_progress_actions,
    parse_contract_response,
    parse_sub_action_id_response,
    parse_sub_response,
    select_candidate_actions,
    remove_blocked_actions,
    snap_action_to_valid,
)
from generate_sft_from_kimi_rollouts import action_id_for_step_action
from rollout_schema import ActionStep


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


def test_parse_sub_action_id_response():
    action_id, done, valid = parse_sub_action_id_response(
        "[action_id]a12[/action_id][subtask_done]false[/subtask_done]"
    )
    assert action_id == "A12"
    assert done is False
    assert valid is True


def test_candidate_action_map_and_lookup():
    candidates = candidate_action_map(
        ["connect x to y", "open cupboard", "look around"],
        max_actions=3,
        include_graph_actions=False,
        context="open cupboard",
    )
    assert candidates[0] == ("A0", "look around")
    assert action_from_id("a1", candidates) == "open cupboard"


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


def test_no_progress_actions_are_removed_when_alternatives_exist():
    blocked = no_progress_actions(
        [
            {
                "action": "look around",
                "reward": 0,
                "score_delta": 0,
                "observation_changed": False,
            }
        ]
    )
    assert blocked == {"look around"}
    assert remove_blocked_actions(["look around", "open cupboard"], blocked) == ["open cupboard"]


def test_repeated_zero_gain_action_is_blocked_even_if_observation_changes():
    blocked = no_progress_actions(
        [
            {
                "action": "pick up chocolate",
                "reward": 0,
                "score_delta": 0,
                "observation_changed": True,
            },
            {
                "action": "pick up chocolate",
                "reward": 0,
                "score_delta": 0,
                "observation_changed": True,
            },
        ]
    )
    assert blocked == {"pick up chocolate"}


def test_unrelated_focus_action_is_ranked_late():
    assert action_rank("focus on drawer", "subgoal: find chocolate")[0] > action_rank(
        "open cupboard",
        "subgoal: find chocolate",
    )[0]


def test_select_candidate_actions_filters_unsafe_focus_actions():
    selected = select_candidate_actions(
        ["focus on stove", "focus on cup containing orange juice", "open cupboard"],
        max_actions=10,
        include_graph_actions=False,
        context="boil orange juice",
    )
    assert "focus on stove" not in selected
    assert "focus on cup containing orange juice" in selected
    assert is_safe_focus_action("focus on cup containing apple juice")
    assert not is_safe_focus_action("focus on freezer")


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


def test_action_id_for_step_action_reads_candidate_list():
    step = ActionStep(
        step_index=0,
        observation="obs",
        raw_response="[action_id]A1[/action_id][subtask_done]false[/subtask_done]",
        action="open cupboard",
        format_valid=True,
        action_valid=True,
        declared_subtask_done=False,
        environment_reward=0,
        score_before=0,
        score_after=0,
        next_observation="obs2",
        environment_done=False,
        prompt_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Candidate actions:\nA0: look around\nA1: open cupboard"},
        ],
    )
    assert action_id_for_step_action(step) == "A1"
