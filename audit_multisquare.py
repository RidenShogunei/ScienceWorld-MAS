"""Audit Multi-Square ScienceWorld trajectories before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scienceworld_data import load_high_trajectories, load_low_trajectories, parse_action_done, task_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    high = load_high_trajectories(data_dir / "expert_high-data.json")
    low = load_low_trajectories(data_dir / "expert_low-data.json")

    malformed_actions = []
    done_mismatches = []
    action_count = 0
    for trajectory in low:
        for step, encoded_action in enumerate(trajectory.actions):
            action_count += 1
            try:
                _, encoded_done = parse_action_done(encoded_action)
            except ValueError as exc:
                malformed_actions.append(str(exc))
                continue
            if encoded_done != trajectory.dones[step]:
                done_mismatches.append({"trajectory": trajectory.source_index, "step": step})

    subtask_count = sum(len(item.subtasks) for item in high)
    report = {
        "high_episodes": len(high),
        "high_subtasks": subtask_count,
        "low_trajectories": len(low),
        "atomic_actions": action_count,
        "subtask_count_delta": subtask_count - len(low),
        "dropped_unlabeled_high_subtasks": sum(
            item.dropped_unlabeled_subtasks for item in high
        ),
        "affected_high_trajectories": sum(
            bool(item.dropped_unlabeled_subtasks) for item in high
        ),
        "unique_task_families": len({task_family(item.task_description) for item in high}),
        "average_subtasks_per_episode": subtask_count / max(len(high), 1),
        "average_actions_per_subtask": action_count / max(len(low), 1),
        "high_with_terminal_observation": sum(
            len(item.observations) == len(item.subtasks) + 1 for item in high
        ),
        "low_with_terminal_observation": sum(
            len(item.observations) == len(item.actions) + 1 for item in low
        ),
        "terminal_high_episodes": sum(bool(item.dones[-1]) for item in high),
        "terminal_low_trajectories": sum(bool(item.dones[-1]) for item in low),
        "malformed_action_count": len(malformed_actions),
        "done_mismatch_count": len(done_mismatches),
        "high_subtask_frequency_top20": Counter(
            subtask for trajectory in high for subtask in trajectory.subtasks
        ).most_common(20),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[audit] wrote {output}")


if __name__ == "__main__":
    main()
