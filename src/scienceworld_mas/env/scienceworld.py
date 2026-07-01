"""Thin wrapper around the official ScienceWorld environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeSpec:
    task_name: str
    variation_id: int
    split: str


@dataclass(frozen=True)
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any]
    action_valid: bool

    @property
    def score(self) -> float:
        return float(self.info.get("score", 0.0))


class ScienceWorldRunner:
    """Small recorded wrapper around `scienceworld.ScienceWorldEnv`.

    The wrapper deliberately exposes official environment score/reward without
    shaping. It also records whether an action was in the environment's valid
    action set before the step was taken; this is diagnostic only.
    """

    def __init__(self, step_limit: int = 100) -> None:
        from scienceworld import ScienceWorldEnv

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
        self.env.load(spec.task_name, spec.variation_id, generateGoldPath=True)
        observation, info = self.env.reset()
        return str(observation), str(self.env.get_task_description()), dict(info)

    def step(self, action: str) -> StepResult:
        valid_actions = set(self.valid_actions())
        observation, reward, done, info = self.env.step(action)
        return StepResult(
            observation=str(observation),
            reward=float(reward),
            done=bool(done),
            info=dict(info),
            action_valid=action in valid_actions,
        )

    def valid_actions(self) -> list[str]:
        return list(self.env.get_valid_action_object_combinations())

    def gold_actions(self) -> list[str]:
        return list(self.env.get_gold_action_sequence())
