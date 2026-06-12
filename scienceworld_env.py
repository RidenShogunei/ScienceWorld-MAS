"""Thin, recorded wrapper around the official ScienceWorld environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scienceworld import ScienceWorldEnv


@dataclass(frozen=True)
class EpisodeSpec:
    task_name: str
    variation_id: int
    split: str


class ScienceWorldRunner:
    def __init__(self, step_limit: int = 100) -> None:
        self.env = ScienceWorldEnv("", envStepLimit=step_limit)
        self.task_names = list(self.env.get_task_names())

    def close(self) -> None:
        self.env.close()

    def variations(self, task_name: str, split: str) -> list[int]:
        self.env.load(task_name)
        if split == "train":
            return list(self.env.get_variations_train())
        if split == "dev":
            return list(self.env.get_variations_dev())
        if split == "test":
            return list(self.env.get_variations_test())
        raise ValueError(f"unknown split: {split}")

    def reset(self, spec: EpisodeSpec) -> tuple[str, str, dict[str, Any]]:
        self.env.load(spec.task_name, spec.variation_id)
        observation, info = self.env.reset()
        return str(observation), str(self.env.get_task_description()), dict(info)

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any], bool]:
        valid_actions = set(self.env.get_valid_action_object_combinations())
        observation, reward, done, info = self.env.step(action)
        return str(observation), float(reward), bool(done), dict(info), action in valid_actions

    def gold_actions(self) -> list[str]:
        return list(self.env.get_gold_action_sequence())

