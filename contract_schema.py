"""Structured Main/Sub communication contract utilities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


CONTRACT_PATTERN = re.compile(r"\[contract\](.*?)\[/contract\]", re.DOTALL)


@dataclass(frozen=True)
class CommunicationContract:
    goal: str
    subgoal: str
    rationale: str
    target_objects: list[str] = field(default_factory=list)
    location_hint: str = ""
    required_tools: list[str] = field(default_factory=list)
    success_condition: str = ""
    action_guidance: list[str] = field(default_factory=list)
    fallback_if_blocked: str = ""

    def validate(self) -> None:
        required = {
            "goal": self.goal,
            "subgoal": self.subgoal,
            "rationale": self.rationale,
            "success_condition": self.success_condition,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"contract missing required fields: {', '.join(missing)}")
        if not self.action_guidance:
            raise ValueError("contract must include at least one action guidance item")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_tagged_json(self) -> str:
        payload = json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True)
        return f"[contract]{payload}[/contract]"


def parse_contract_text(text: str) -> CommunicationContract:
    match = CONTRACT_PATTERN.search(text)
    payload_text = match.group(1).strip() if match else text.strip()
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("contract payload must be a JSON object")
    contract = CommunicationContract(
        goal=str(payload.get("goal", "")).strip(),
        subgoal=str(payload.get("subgoal", "")).strip(),
        rationale=str(payload.get("rationale", "")).strip(),
        target_objects=[str(item).strip() for item in payload.get("target_objects", []) if str(item).strip()],
        location_hint=str(payload.get("location_hint", "")).strip(),
        required_tools=[str(item).strip() for item in payload.get("required_tools", []) if str(item).strip()],
        success_condition=str(payload.get("success_condition", "")).strip(),
        action_guidance=[
            str(item).strip() for item in payload.get("action_guidance", []) if str(item).strip()
        ],
        fallback_if_blocked=str(payload.get("fallback_if_blocked", "")).strip(),
    )
    contract.validate()
    return contract


def build_mock_contract(
    *,
    task: str,
    subtask: str,
    expert_actions: list[str],
    observation: str,
) -> CommunicationContract:
    """Deterministic fallback used for tests and dry runs."""
    action_guidance = expert_actions[:4] if expert_actions else ["inspect the current observation"]
    target_objects = _guess_target_objects(subtask, action_guidance)
    return CommunicationContract(
        goal=task,
        subgoal=subtask,
        rationale="This subgoal is the next expert step toward solving the ScienceWorld task.",
        target_objects=target_objects,
        location_hint=_guess_location(observation, action_guidance),
        required_tools=_guess_tools(action_guidance),
        success_condition=f"The subgoal is complete when the executor has achieved: {subtask}.",
        action_guidance=action_guidance,
        fallback_if_blocked="Look around, inspect inventory, and choose a valid action involving visible objects.",
    )


def _guess_location(observation: str, actions: list[str]) -> str:
    for line in observation.splitlines():
        line = line.strip()
        if line.lower().startswith("this room is called"):
            return line
    for action in actions:
        if " to " in action:
            return action
    return ""


def _guess_tools(actions: list[str]) -> list[str]:
    tools = []
    for action in actions:
        for marker in ("thermometer", "metal pot", "jug", "shovel", "beaker"):
            if marker in action.lower() and marker not in tools:
                tools.append(marker)
    return tools


def _guess_target_objects(subtask: str, actions: list[str]) -> list[str]:
    candidates = []
    text = " ".join([subtask, *actions]).lower()
    for marker in (
        "kitchen",
        "door",
        "substance",
        "metal pot",
        "thermometer",
        "stove",
        "sink",
        "plant",
        "animal",
        "paint",
    ):
        if marker in text:
            candidates.append(marker)
    return candidates

