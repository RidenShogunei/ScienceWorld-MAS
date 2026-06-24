from evaluate_environment import main_messages, sub_messages
from generate_minimal_contract_sft_data import MINIMAL_MAIN_SYSTEM, MINIMAL_SUB_SYSTEM


def test_contract_simple_main_prompt_matches_training_shape():
    messages = main_messages(
        "Boil water",
        "This room is called the hallway.",
        ["look around"],
        "contract-simple",
    )
    assert messages[0] == {"role": "system", "content": MINIMAL_MAIN_SYSTEM}
    assert messages[1]["content"] == (
        "Task:\nBoil water\n\nPlanner state:\n"
        "Group action:['look around']. Current observation: "
        "This room is called the hallway."
    )


def test_contract_simple_sub_prompt_has_no_history_or_valid_actions():
    contract = '[contract]{"subgoal":"Navigate to kitchen"}[/contract]'
    messages = sub_messages(
        contract,
        "The kitchen door is closed.",
        "contract-simple",
    )
    assert messages[0] == {"role": "system", "content": MINIMAL_SUB_SYSTEM}
    assert messages[1]["content"] == (
        f"Contract:\n{contract}\n\nObservation:\nThe kitchen door is closed."
    )
    assert "Recent execution history:" not in messages[1]["content"]
    assert "Valid actions:" not in messages[1]["content"]
