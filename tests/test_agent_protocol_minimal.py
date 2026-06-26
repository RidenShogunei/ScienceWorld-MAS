from agent_protocol import parse_main_output

MINIMAL_JSON = (
    '[contract]{"subgoal":"Navigate to kitchen","success_condition":"The agent is in the kitchen.",'
    '"target_objects":["kitchen"],"action_guidance":["go to kitchen"],'
    '"handoff_if":"complete when success_condition is met; need_replan if blocked"}[/contract]'
)


def test_parse_main_output_minimal_uses_minimal_parser():
    contract = parse_main_output("minimal", MINIMAL_JSON)
    assert contract is not None
    assert contract.subgoal == "Navigate to kitchen"
    assert contract.action_guidance == ["go to kitchen"]


def test_parse_main_output_contract_rejects_minimal_json():
    assert parse_main_output("contract", MINIMAL_JSON) is None
