"""CLI for strict pass@1 ScienceWorld evaluation."""

from __future__ import annotations

import argparse
import json

from scienceworld_mas.agents import FirstValidActionPolicy, GoldActionPolicy, load_hierarchical_hf_policy
from scienceworld_mas.env import (
    ScienceWorldRunner,
    episode_list_metadata,
    generate_stratified_episodes,
    load_episode_list,
    save_episode_list,
)
from scienceworld_mas.evaluation import evaluate_episodes, write_evaluation_report


def build_policy(args):
    if isinstance(args, str):
        name = args
    else:
        name = args.policy
    if name == "gold":
        return GoldActionPolicy()
    if name == "first-valid":
        return FirstValidActionPolicy()
    if name == "hf-hierarchical":
        missing = [
            option
            for option in ("base_model", "system1_adapter", "system2_adapter")
            if not getattr(args, option, None)
        ]
        if missing:
            raise ValueError(
                "hf-hierarchical policy requires: "
                + ", ".join(f"--{item.replace('_', '-')}" for item in missing)
            )
        return load_hierarchical_hf_policy(
            base_model=args.base_model,
            system1_adapter=args.system1_adapter,
            system2_adapter=args.system2_adapter,
            use_4bit=args.use_4bit,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            system1_max_new_tokens=args.system1_max_new_tokens,
            system2_max_new_tokens=args.system2_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    raise ValueError(f"unknown policy: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("gold", "first-valid", "hf-hierarchical"), default="gold")
    parser.add_argument("--episode-list", default=None)
    parser.add_argument("--write-episode-list", default=None)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--k-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--output", default="artifacts/eval/v2_strict_pass1.json")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--system1-adapter", default=None)
    parser.add_argument("--system2-adapter", default=None)
    parser.add_argument("--system1-max-new-tokens", type=int, default=64)
    parser.add_argument("--system2-max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
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
            build_policy(args),
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
