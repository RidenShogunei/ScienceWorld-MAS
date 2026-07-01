"""Convert v2 transitions into supervised chat examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


RoleName = Literal["system1", "system2"]


SYSTEM1_SYSTEM_PROMPT = (
    "You are System1, the ScienceWorld high-level planner. Given the task and "
    "current observation, output the next concise subgoal for System2."
)


SYSTEM2_SYSTEM_PROMPT = (
    "You are System2, the ScienceWorld executor. Given a subgoal and current "
    "observation, output one executable ScienceWorld action and whether the "
    "subgoal is complete."
)


@dataclass(frozen=True)
class TrainingExample:
    role: RoleName
    messages: tuple[dict[str, str], ...]
    source: dict

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "messages": list(self.messages),
            "source": self.source,
        }


def _require(mapping: dict, key: str) -> str:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"missing required transition field: {key}")
    return str(value)


def transition_to_example(transition: dict) -> TrainingExample:
    role = transition.get("role")
    if role == "system1":
        task = _require(transition, "task_description")
        observation = _require(transition, "observation")
        target = _require(transition, "target_subgoal")
        return TrainingExample(
            role="system1",
            messages=(
                {"role": "system", "content": SYSTEM1_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Task:\n{task}\n\nObservation:\n{observation}",
                },
                {"role": "assistant", "content": target},
            ),
            source=transition,
        )

    if role == "system2":
        subgoal = _require(transition, "subgoal")
        observation = _require(transition, "observation")
        action = _require(transition, "target_action")
        done = bool(transition.get("subgoal_done", False))
        return TrainingExample(
            role="system2",
            messages=(
                {"role": "system", "content": SYSTEM2_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Subgoal:\n{subgoal}\n\nObservation:\n{observation}",
                },
                {
                    "role": "assistant",
                    "content": (
                        f"[action]{action}[/action]"
                        f"[subgoal_done]{str(done).lower()}[/subgoal_done]"
                    ),
                },
            ),
            source=transition,
        )

    raise ValueError(f"unknown transition role: {role!r}")


def load_transition_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def load_training_examples(data_dir: str | Path, *, role: RoleName, split: str) -> list[TrainingExample]:
    path = Path(data_dir) / role / f"{split}.jsonl"
    return [transition_to_example(record) for record in load_transition_jsonl(path)]
