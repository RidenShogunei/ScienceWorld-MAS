"""Build System1/System2 transition datasets from Multi-Square trajectories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .multisquare import HighTrajectory, LowTrajectory
from .splits import assign_split, normalize_group_text, strip_embedded_instruction, task_family


@dataclass(frozen=True)
class System1Transition:
    task_description: str
    observation: str
    target_subgoal: str
    reward: float
    score: float
    done: bool
    source_index: int
    trajectory_step: int
    split_group: str

    def to_dict(self) -> dict:
        return asdict(self) | {"role": "system1"}


@dataclass(frozen=True)
class System2Transition:
    subgoal: str
    observation: str
    target_action: str
    subgoal_done: bool
    reward: float
    score: float
    episode_done: bool
    source_index: int
    trajectory_step: int
    split_group: str

    def to_dict(self) -> dict:
        return asdict(self) | {"role": "system2"}


@dataclass(frozen=True)
class DatasetManifest:
    seed: int
    train_ratio: float
    val_ratio: float
    high_trajectories: int
    low_trajectories: int
    system1_counts: dict[str, int]
    system2_counts: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SplitDatasets:
    system1: dict[str, list[System1Transition]]
    system2: dict[str, list[System2Transition]]
    manifest: DatasetManifest


def parse_action_done(value: str) -> tuple[str, bool]:
    action, separator, done_text = value.rpartition(";")
    if not separator or done_text.strip().lower() not in {"true", "false"}:
        raise ValueError(f"invalid Multi-Square action/done value: {value!r}")
    return action.strip(), done_text.strip().lower() == "true"


def high_to_transitions(trajectory: HighTrajectory) -> list[System1Transition]:
    task = strip_embedded_instruction(trajectory.task_description, "Task Description:")
    split_group = task_family(trajectory.task_description)
    return [
        System1Transition(
            task_description=task,
            observation=trajectory.observations[step],
            target_subgoal=trajectory.subtasks[step].strip(),
            reward=trajectory.rewards[step],
            score=trajectory.scores[step],
            done=trajectory.dones[step],
            source_index=trajectory.source_index,
            trajectory_step=step,
            split_group=split_group,
        )
        for step in range(len(trajectory.subtasks))
    ]


def low_to_transitions(trajectory: LowTrajectory) -> list[System2Transition]:
    subgoal = strip_embedded_instruction(trajectory.subtask_prompt, "Subtask:")
    split_group = f"system2:{normalize_group_text(subgoal)}"
    transitions = []
    for step, encoded_action in enumerate(trajectory.actions):
        action, subgoal_done = parse_action_done(encoded_action)
        transitions.append(
            System2Transition(
                subgoal=subgoal,
                observation=trajectory.observations[step],
                target_action=action,
                subgoal_done=subgoal_done,
                reward=trajectory.rewards[step],
                score=trajectory.scores[step],
                episode_done=trajectory.dones[step],
                source_index=trajectory.source_index,
                trajectory_step=step,
                split_group=split_group,
            )
        )
    return transitions


def _empty_splits() -> dict[str, list]:
    return {"train": [], "val": [], "test": []}


def build_transition_datasets(
    high: list[HighTrajectory],
    low: list[LowTrajectory],
    *,
    seed: int = 123,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> SplitDatasets:
    system1 = _empty_splits()
    system2 = _empty_splits()

    for trajectory in high:
        transitions = high_to_transitions(trajectory)
        if not transitions:
            continue
        split = assign_split(
            transitions[0].split_group,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        system1[split].extend(transitions)

    for trajectory in low:
        transitions = low_to_transitions(trajectory)
        if not transitions:
            continue
        split = assign_split(
            transitions[0].split_group,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        system2[split].extend(transitions)

    manifest = DatasetManifest(
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        high_trajectories=len(high),
        low_trajectories=len(low),
        system1_counts={split: len(items) for split, items in system1.items()},
        system2_counts={split: len(items) for split, items in system2.items()},
    )
    return SplitDatasets(system1=system1, system2=system2, manifest=manifest)


def write_transition_datasets(datasets: SplitDatasets, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for role_name, splits in (("system1", datasets.system1), ("system2", datasets.system2)):
        role_dir = output / role_name
        role_dir.mkdir(parents=True, exist_ok=True)
        for split, items in splits.items():
            with (role_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(datasets.manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
