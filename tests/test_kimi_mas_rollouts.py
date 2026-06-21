from collect_kimi_mas_rollouts import (
    action_rank,
    parse_contract_response,
    parse_minimal_contract_response,
    parse_sub_response,
    parse_args as parse_collect_args,
    select_candidate_actions,
)
from generate_sft_from_kimi_rollouts import parse_args as parse_sft_args


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
        '\u2022 ```json\n{"goal":"g","subgoal":"s","rationale":"r",'
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


def test_parse_minimal_contract_response_from_tagged_json():
    contract = parse_minimal_contract_response(
        '[contract]{"subgoal":"reach kitchen","success_condition":"agent is in kitchen",'
        '"target_objects":["door to kitchen"],"action_guidance":["open door to kitchen"],'
        '"handoff_if":"complete when success_condition is met; need_replan if blocked"}[/contract]'
    )
    assert contract is not None
    assert contract.subgoal == "reach kitchen"
    assert contract.action_guidance == ["open door to kitchen"]


def test_parse_sub_response():
    action, done, handoff, valid = parse_sub_response(
        "[action]open door to kitchen[/action][subtask_done]true[/subtask_done][handoff]complete[/handoff]"
    )
    assert action == "open door to kitchen"
    assert done is True
    assert handoff == "complete"
    assert valid is True


def test_parse_sub_response_defaults_handoff_for_legacy_output():
    action, done, handoff, valid = parse_sub_response(
        "[action]open door to kitchen[/action][subtask_done]false[/subtask_done]"
    )
    assert action == "open door to kitchen"
    assert done is False
    assert handoff == "continue"
    assert valid is True


def test_parse_sub_response_invalid():
    action, done, handoff, valid = parse_sub_response("garbage text without tags")
    assert action is None
    assert done is False
    assert handoff == "continue"
    assert valid is False


def test_select_candidate_actions_preserves_environment_actions_by_default():
    actions = [
        "connect door to hallway",
        "focus on drawer",
        "look around",
        "focus on inclined plane with a copper surface",
    ]
    assert select_candidate_actions(actions) == actions


def test_select_candidate_actions_can_rank_by_context():
    selected = select_candidate_actions(
        ["focus on stove", "open cupboard", "go to kitchen", "pick up chocolate"],
        max_actions=10,
        rank_actions=True,
        context="Need to open the cupboard and pick up chocolate.",
    )
    assert selected[0] in {"open cupboard", "go to kitchen", "pick up chocolate"}
    assert selected[-1] == "focus on stove"


def test_unrelated_focus_action_is_ranked_late():
    result = action_rank("focus on drawer", "subgoal: find chocolate")
    assert result == action_rank("focus on drawer", "subgoal: find chocolate")


def test_native_rollout_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["collect_kimi_mas_rollouts.py"])
    args = parse_collect_args()
    assert args.model == "kimi-for-coding"
    assert args.api_base == "https://api.kimi.com/coding"
    assert args.api_key_env == "KIMI_CODE_API_KEY"
    assert args.max_valid_actions == 0
    assert args.max_steps_per_subtask == 6
    assert args.rank_valid_actions is False
    assert args.contract_schema == "verbose"


def test_kimi_sft_conversion_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["generate_sft_from_kimi_rollouts.py", "--input", "rollouts.jsonl"],
    )
    args = parse_sft_args()
    assert args.valid_actions_only is True
    assert args.success_only is False
    assert args.keep_local_nonnegative_steps is False
    assert args.contract_schema == "auto"
