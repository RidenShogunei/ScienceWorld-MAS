"""Load, validate, and split Multi-Square ScienceWorld expert trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HIGH_KEYS = ("task_description", "obs", "subtask", "reward", "score", "done")
LOW_KEYS = ("subtask", "obs", "action", "reward", "score", "done")


@dataclass(frozen=True)
class HighTrajectory:
    source_index: int
    task_description: str
    observations: list[str]
    subtasks: list[str]
    rewards: list[float]
    scores: list[float]
    dones: list[bool]
    dropped_unlabeled_subtasks: int = 0


@dataclass(frozen=True)
class LowTrajectory:
    source_index: int
    subtask_prompt: str
    observations: list[str]
    actions: list[str]
    rewards: list[float]
    scores: list[float]
    dones: list[bool]


def load_columnar_json(path: str | Path, required_keys: Iterable[str]) -> dict[str, list[Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(missing)}")

    lengths = {key: len(data[key]) for key in required_keys if isinstance(data[key], list)}
    if len(lengths) != len(tuple(required_keys)):
        raise ValueError(f"{path} columns must all be JSON arrays")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path} columns have inconsistent lengths: {lengths}")
    return data


def _validate_parallel_lists(label: str, index: int, fields: dict[str, list[Any]]) -> None:
    lengths = {name: len(values) for name, values in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{label}[{index}] has misaligned trajectory fields: {lengths}")
    if not next(iter(lengths.values()), 0):
        raise ValueError(f"{label}[{index}] is empty")


def load_high_trajectories(path: str | Path) -> list[HighTrajectory]:
    data = load_columnar_json(path, HIGH_KEYS)
    trajectories = []
    for index in range(len(data["task_description"])):
        subtasks = [str(value) for value in data["subtask"][index]]
        rewards = [float(value) for value in data["reward"][index]]
        scores = [float(value) for value in data["score"][index]]
        dones = [bool(value) for value in data["done"][index]]
        fields = {
            "reward": rewards,
            "score": scores,
            "done": dones,
        }
        _validate_parallel_lists("high", index, fields)
        observations = [str(value) for value in data["obs"][index]]
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
                observations=observations,
                subtasks=subtasks[:labeled_step_count],
                rewards=rewards,
                scores=scores,
                dones=dones,
                dropped_unlabeled_subtasks=dropped_unlabeled,
            )
        )
    return trajectories


def load_low_trajectories(path: str | Path) -> list[LowTrajectory]:
    data = load_columnar_json(path, LOW_KEYS)
    trajectories = []
    for index in range(len(data["subtask"])):
        fields = {
            "action": data["action"][index],
            "reward": data["reward"][index],
            "score": data["score"][index],
            "done": data["done"][index],
        }
        _validate_parallel_lists("low", index, fields)

        observations = [str(value) for value in data["obs"][index]]
        actions = [str(value) for value in fields["action"]]
        if len(observations) not in (len(actions), len(actions) + 1):
            raise ValueError(
                f"low[{index}] observations/actions are incompatible: "
                f"{len(observations)} observations, {len(actions)} actions"
            )

        trajectories.append(
            LowTrajectory(
                source_index=index,
                subtask_prompt=str(data["subtask"][index]),
                observations=observations,
                actions=actions,
                rewards=[float(value) for value in fields["reward"]],
                scores=[float(value) for value in fields["score"]],
                dones=[bool(value) for value in fields["done"]],
            )
        )
    return trajectories


def strip_embedded_instruction(text: str, marker: str) -> str:
    """Remove the dataset's role prompt while retaining the task after its marker."""
    position = text.rfind(marker)
    if position < 0:
        return text.strip()
    return text[position + len(marker) :].strip()


def task_family(task_description: str) -> str:
    """Return a stable grouping key used to avoid random step-level leakage."""
    task = strip_embedded_instruction(task_description, "Task Description:")
    task = task.lower()
    task = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", task)
    task = re.sub(r"\s+", " ", task).strip()
    return task


def assign_split(group_key: str, seed: int = 123, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("ratios must satisfy train > 0, val >= 0, and train + val < 1")
    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def parse_action_done(value: str) -> tuple[str, bool]:
    action, separator, done_text = value.rpartition(";")
    if not separator or done_text.strip().lower() not in {"true", "false"}:
        raise ValueError(f"invalid Multi-Square action/done value: {value!r}")
    return action.strip(), done_text.strip().lower() == "true"
