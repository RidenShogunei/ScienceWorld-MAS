"""Load Multi-Square ScienceWorld expert trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HIGH_KEYS = ("task_description", "obs", "subtask", "reward", "score", "done")
LOW_KEYS = ("subtask", "obs", "action", "reward", "score", "done")


@dataclass(frozen=True)
class HighTrajectory:
    source_index: int
    task_description: str
    observations: tuple[str, ...]
    subtasks: tuple[str, ...]
    rewards: tuple[float, ...]
    scores: tuple[float, ...]
    dones: tuple[bool, ...]
    dropped_unlabeled_subtasks: int = 0


@dataclass(frozen=True)
class LowTrajectory:
    source_index: int
    subtask_prompt: str
    observations: tuple[str, ...]
    actions: tuple[str, ...]
    rewards: tuple[float, ...]
    scores: tuple[float, ...]
    dones: tuple[bool, ...]


def load_columnar_json(path: str | Path, required_keys: Iterable[str]) -> dict[str, list[Any]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{input_path} must contain a JSON object")

    keys = tuple(required_keys)
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{input_path} is missing keys: {', '.join(missing)}")

    lengths = {}
    for key in keys:
        if not isinstance(data[key], list):
            raise ValueError(f"{input_path} column {key!r} must be a JSON array")
        lengths[key] = len(data[key])
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{input_path} columns have inconsistent lengths: {lengths}")
    return data


def _validate_parallel_lists(label: str, index: int, fields: dict[str, list[Any]]) -> None:
    lengths = {name: len(values) for name, values in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{label}[{index}] has misaligned trajectory fields: {lengths}")
    if not next(iter(lengths.values()), 0):
        raise ValueError(f"{label}[{index}] is empty")


def load_high_trajectories(path: str | Path) -> list[HighTrajectory]:
    data = load_columnar_json(path, HIGH_KEYS)
    trajectories: list[HighTrajectory] = []
    for index in range(len(data["task_description"])):
        subtasks = [str(value) for value in data["subtask"][index]]
        observations = [str(value) for value in data["obs"][index]]
        rewards = [float(value) for value in data["reward"][index]]
        scores = [float(value) for value in data["score"][index]]
        dones = [bool(value) for value in data["done"][index]]
        _validate_parallel_lists(
            "high",
            index,
            {"reward": rewards, "score": scores, "done": dones},
        )

        labeled_step_count = len(rewards)
        if len(subtasks) < labeled_step_count:
            raise ValueError(
                f"high[{index}] has fewer subtasks than reward labels: "
                f"{len(subtasks)} subtasks, {labeled_step_count} labels"
            )
        if len(observations) < labeled_step_count:
            raise ValueError(
                f"high[{index}] has fewer observations than reward labels: "
                f"{len(observations)} observations, {labeled_step_count} labels"
            )
        dropped_unlabeled = len(subtasks) - labeled_step_count
        if dropped_unlabeled not in (0, 1):
            raise ValueError(
                f"high[{index}] has unsupported unlabeled subtask count: {dropped_unlabeled}"
            )
        if len(observations) not in (labeled_step_count, labeled_step_count + 1):
            raise ValueError(
                f"high[{index}] observations/labels are incompatible: "
                f"{len(observations)} observations, {labeled_step_count} labels"
            )

        trajectories.append(
            HighTrajectory(
                source_index=index,
                task_description=str(data["task_description"][index]),
                observations=tuple(observations),
                subtasks=tuple(subtasks[:labeled_step_count]),
                rewards=tuple(rewards),
                scores=tuple(scores),
                dones=tuple(dones),
                dropped_unlabeled_subtasks=dropped_unlabeled,
            )
        )
    return trajectories


def load_low_trajectories(path: str | Path) -> list[LowTrajectory]:
    data = load_columnar_json(path, LOW_KEYS)
    trajectories: list[LowTrajectory] = []
    for index in range(len(data["subtask"])):
        raw_actions = data["action"][index]
        rewards = [float(value) for value in data["reward"][index]]
        scores = [float(value) for value in data["score"][index]]
        dones = [bool(value) for value in data["done"][index]]
        _validate_parallel_lists(
            "low",
            index,
            {"action": raw_actions, "reward": rewards, "score": scores, "done": dones},
        )

        observations = [str(value) for value in data["obs"][index]]
        actions = [str(value) for value in raw_actions]
        if len(observations) not in (len(actions), len(actions) + 1):
            raise ValueError(
                f"low[{index}] observations/actions are incompatible: "
                f"{len(observations)} observations, {len(actions)} actions"
            )

        trajectories.append(
            LowTrajectory(
                source_index=index,
                subtask_prompt=str(data["subtask"][index]),
                observations=tuple(observations),
                actions=tuple(actions),
                rewards=tuple(rewards),
                scores=tuple(scores),
                dones=tuple(dones),
            )
        )
    return trajectories
