"""Action-ID Sub protocol (minimal copy for L1)."""

from __future__ import annotations

import json
import re
from typing import Any

from collect_kimi_mas_rollouts import select_candidate_actions

ACTION_ID_SUB_SYSTEM = (
    "You are the ScienceWorld executor. Choose exactly one candidate action by its integer ID. "
    "Output exactly one line in this format:\n"
    "selected_action_id: <integer>\n"
    "Use only IDs listed under Candidate actions. Do not output free-form action text."
)

ACTION_ID_PATTERN = re.compile(
    r"selected_action_id\s*:\s*(\d+)",
    re.IGNORECASE,
)


def extract_inventory(observation: str) -> str:
    marker = "In your inventory, you see:"
    if marker not in observation:
        return "(empty)"
    tail = observation.split(marker, 1)[1]
    for stop in ("\nYou also see:", "\nYour current task", "\nTask Description:"):
        if stop in tail:
            tail = tail.split(stop, 1)[0]
    return tail.strip() or "(empty)"


def build_action_id_messages(
    *,
    task: str,
    observation: str,
    candidate_actions: list[str],
    recent_history: list[dict[str, Any]] | None = None,
    contract: str | None = None,
) -> list[dict[str, str]]:
    parts = [
        f"Task:\n{task.strip()}",
        f"Observation:\n{observation.strip()}",
        f"Inventory:\n{extract_inventory(observation)}",
    ]
    if recent_history:
        parts.append(
            "Recent history:\n"
            + json.dumps(recent_history, ensure_ascii=False, indent=2)
        )
    numbered = "\n".join(f"{index}: {action}" for index, action in enumerate(candidate_actions))
    parts.append(f"Candidate actions:\n{numbered}")
    if contract:
        parts.append(f"Contract:\n{contract.strip()}")
    return [
        {"role": "system", "content": ACTION_ID_SUB_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def rank_candidate_actions(
    valid_actions: list[str],
    *,
    context: str = "",
    max_actions: int = 32,
) -> list[str]:
    return select_candidate_actions(
        valid_actions,
        max_actions=max_actions,
        rank_actions=True,
        context=context,
    )


def expert_action_id(expert_action: str, candidate_actions: list[str]) -> int | None:
    if expert_action in candidate_actions:
        return candidate_actions.index(expert_action)
    lowered = expert_action.strip().lower()
    for index, action in enumerate(candidate_actions):
        if action.strip().lower() == lowered:
            return index
    return None


def format_assistant_action_id(action_id: int) -> str:
    return f"selected_action_id: {action_id}"


def parse_action_id_response(text: str) -> int | None:
    match = ACTION_ID_PATTERN.search(text.strip())
    if not match:
        return None
    return int(match.group(1))


def decode_action_id(action_id: int | None, candidate_actions: list[str]) -> str | None:
    if action_id is None or action_id < 0 or action_id >= len(candidate_actions):
        return None
    return candidate_actions[action_id]
