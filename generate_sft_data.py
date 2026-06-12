"""Convert Multi-Square ScienceWorld expert trajectories into Main/Sub chat SFT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scienceworld_data import (
    assign_split,
    load_high_trajectories,
    load_low_trajectories,
    parse_action_done,
    strip_embedded_instruction,
    task_family,
)


MAIN_SYSTEM = (
    "You are the main planning agent in ScienceWorld. Given the task, current observation, "
    "and actions completed since the previous plan, choose one clear next subtask for the "
    "executor. Output exactly: [subtask]...[/subtask]"
)

SUB_SYSTEM = (
    "You are the ScienceWorld action executor. Given a subtask and the current observation, "
    "produce one executable environment action and state whether the subtask is complete. "
    "Output exactly: [action]...[/action][subtask_done]true|false[/subtask_done]"
)


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def high_samples(trajectory) -> list[dict]:
    task = strip_embedded_instruction(trajectory.task_description, "Task Description:")
    samples = []
    for step, (observation, subtask) in enumerate(zip(trajectory.observations, trajectory.subtasks)):
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": MAIN_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Task:\n{task}\n\nPlanner state:\n{observation}",
                    },
                    {"role": "assistant", "content": f"[subtask]{subtask.strip()}[/subtask]"},
                ],
                "category": "main",
                "stage": "plan",
                "source_index": trajectory.source_index,
                "trajectory_step": step,
                "task_family": task_family(trajectory.task_description),
                "environment_reward": trajectory.rewards[step],
                "environment_score": trajectory.scores[step],
                "episode_done": trajectory.dones[step],
            }
        )
    return samples


def low_samples(trajectory) -> list[dict]:
    subtask = strip_embedded_instruction(trajectory.subtask_prompt, "Subtask:")
    samples = []
    for step, encoded_action in enumerate(trajectory.actions):
        action, encoded_done = parse_action_done(encoded_action)
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": SUB_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Subtask:\n{subtask}\n\nObservation:\n{trajectory.observations[step]}",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"[action]{action}[/action]"
                            f"[subtask_done]{str(encoded_done).lower()}[/subtask_done]"
                        ),
                    },
                ],
                "category": "sub",
                "stage": "act",
                "source_index": trajectory.source_index,
                "trajectory_step": step,
                "environment_reward": trajectory.rewards[step],
                "environment_score": trajectory.scores[step],
                "subtask_done": trajectory.dones[step],
            }
        )
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit-high", type=int, default=None)
    parser.add_argument("--limit-low", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    high = load_high_trajectories(data_dir / "expert_high-data.json")
    low = load_low_trajectories(data_dir / "expert_low-data.json")
    if args.limit_high is not None:
        high = high[: args.limit_high]
    if args.limit_low is not None:
        low = low[: args.limit_low]

    by_split = {split: [] for split in ("train", "val", "test")}
    high_split_counts = Counter()
    for trajectory in high:
        split = assign_split(
            task_family(trajectory.task_description),
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        samples = high_samples(trajectory)
        by_split[split].extend(samples)
        high_split_counts[split] += len(samples)

    # Low-level records do not contain the parent task. They are split independently by
    # normalized subtask text, preventing identical executor objectives crossing splits.
    low_split_counts = Counter()
    for trajectory in low:
        subtask = strip_embedded_instruction(trajectory.subtask_prompt, "Subtask:").lower()
        split = assign_split(
            f"low:{subtask}",
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        samples = low_samples(trajectory)
        by_split[split].extend(samples)
        low_split_counts[split] += len(samples)

    for split, samples in by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", samples)

    manifest = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "high_trajectories": len(high),
        "low_trajectories": len(low),
        "samples": {split: len(samples) for split, samples in by_split.items()},
        "main_samples": dict(high_split_counts),
        "sub_samples": dict(low_split_counts),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

