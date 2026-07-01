"""Main contract prompts and parsing for L1."""

from __future__ import annotations

from collect_kimi_mas_rollouts import parse_minimal_contract_response
from generate_minimal_contract_sft_data import MINIMAL_MAIN_SYSTEM, MinimalContract


def main_messages(
    task: str,
    observation: str,
    group_actions: list[str] | None = None,
) -> list[dict[str, str]]:
    actions = group_actions or []
    state = f"Group action:{actions}. Current observation: {observation}"
    return [
        {"role": "system", "content": MINIMAL_MAIN_SYSTEM},
        {"role": "user", "content": f"Task:\n{task}\n\nPlanner state:\n{state}"},
    ]


def parse_main_contract(text: str) -> MinimalContract | None:
    return parse_minimal_contract_response(text)


def contract_text_for_sub(text: str) -> tuple[str, bool]:
    contract = parse_main_contract(text)
    if contract is None:
        return text.strip(), False
    return contract.to_tagged_json(), True
