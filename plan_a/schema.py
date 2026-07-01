"""Plan A contract: Main writes subgoal + focus_objects only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PLAN_KEYS = ("subgoal", "focus_objects")

PLAN_A_MAIN_SYSTEM = (
    "You are the main planning agent in ScienceWorld. Output exactly one "
    "[plan]{...}[/plan] block containing JSON with keys: subgoal, focus_objects. "
    "subgoal is one concise sentence for the current planning step. "
    "focus_objects is a JSON array of relevant object or location strings."
)

PLAN_A_SUB_SYSTEM = (
    "You are the ScienceWorld executor. Read the subgoal, focus objects, task, "
    "observation, and numbered candidate actions. Choose exactly one candidate "
    "action by its integer ID. Output exactly one line:\n"
    "selected_action_id: <integer>\n"
    "Use only IDs listed under Candidate actions."
)


@dataclass(frozen=True)
class PlanContract:
    subgoal: str
    focus_objects: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "subgoal": self.subgoal.strip(),
            "focus_objects": [item.strip() for item in self.focus_objects if item.strip()],
        }

    def to_tagged_json(self) -> str:
        payload = self.to_payload()
        return (
            "[plan]"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "[/plan]"
        )

    def rank_context(self) -> str:
        payload = self.to_payload()
        return json.dumps(payload, ensure_ascii=False)

    def subgoal_block(self) -> str:
        objects = ", ".join(self.to_payload()["focus_objects"]) or "(none)"
        return f"Subgoal:\n{self.subgoal.strip()}\n\nFocus objects:\n{objects}"


def parse_plan_response(text: str) -> PlanContract | None:
    from generate_minimal_contract_sft_data import extract_first_json_object

    raw = text.strip()
    if "[plan]" in raw.lower():
        start = raw.lower().find("[plan]")
        end = raw.lower().find("[/plan]")
        if end > start:
            raw = raw[start + len("[plan]") : end].strip()
        else:
            raw = raw.split("[plan]", 1)[-1].strip()
    try:
        if raw.startswith("{"):
            payload = json.loads(raw)
        else:
            payload = json.loads(extract_first_json_object(raw))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    subgoal = str(payload.get("subgoal", "")).strip()
    focus_raw = payload.get("focus_objects", [])
    if not subgoal:
        return None
    if isinstance(focus_raw, str):
        focus_objects = [part.strip() for part in focus_raw.split(",") if part.strip()]
    elif isinstance(focus_raw, list):
        focus_objects = [str(item).strip() for item in focus_raw if str(item).strip()]
    else:
        focus_objects = []
    return PlanContract(subgoal=subgoal, focus_objects=focus_objects)
