"""CLI for strict pass@1 ScienceWorld evaluation."""

from __future__ import annotations

import argparse
import json

from scienceworld_mas.agents import FirstValidActionPolicy, GoldActionPolicy
from scienceworld_mas.env import (
    ScienceWorldRunner,
    episode_list_metadata,
    generate_stratified_episodes,
    load_episode_list,
    save_episode_list,
)
from scienceworld_mas.evaluation import evaluate_episodes, write_evaluation_report


def build_policy(name: str):
    if name == "gold":
        return GoldActionPolicy()
    if name == "first-valid":
        return FirstValidActionPolicy()
    raise ValueError(f"unknown policy: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("gold", "first-valid"), default="gold")
    parser.add_argument("--episode-list", default=None)
    parser.add_argument("--write-episode-list", default=None)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--k-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--output", default="artifacts/eval/v2_strict_pass1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ScienceWorldRunner(step_limit=args.step_limit)
    try:
        if args.episode_list:
            metadata, specs = load_episode_list(args.episode_list)
        else:
            specs = generate_stratified_episodes(
                runner,
                split=args.split,
                k_per_task=args.k_per_task,
                seed=args.seed,
                task_names=args.tasks,
            )
            metadata = episode_list_metadata(
                specs,
                split=args.split,
                seed=args.seed,
                k_per_task=args.k_per_task,
            )
            if args.write_episode_list:
                save_episode_list(args.write_episode_list, specs, metadata)

        report = evaluate_episodes(
            runner,
            build_policy(args.policy),
            specs,
            step_limit=args.step_limit,
            episode_list=metadata,
        )
    finally:
        runner.close()

    write_evaluation_report(report, args.output)
    print(json.dumps(report.metrics.to_dict(), ensure_ascii=False, indent=2))
    print(f"[eval] wrote {args.output}")


if __name__ == "__main__":
    main()
