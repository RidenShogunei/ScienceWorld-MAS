"""Generate Plan A SFT data from Multi-Square expert trajectories.

Main labels: subgoal + focus_objects only.
Sub labels: action-id selection against candidate actions derived from expert chunk.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from generate_contract_sft_data import ExpertStep, iter_expert_steps
from generate_expert_subtask_contract_sft import (
    explicit_target_conflict,
    recent_history,
)
from generate_minimal_contract_sft_data import clean_contract_text, extract_target_objects
from generate_sft_data import write_jsonl
from l1.protocol import (
    build_action_id_messages,
    expert_action_id,
    format_assistant_action_id,
    rank_candidate_actions,
)

from plan_a.schema import PLAN_A_MAIN_SYSTEM, PLAN_A_SUB_SYSTEM, PlanContract


def plan_for_step(step: ExpertStep, *, focus_limit: int) -> PlanContract:
    return PlanContract(
        subgoal=clean_contract_text(step.subtask.strip()),
        focus_objects=extract_target_objects(step.subtask, step.expert_actions)[:focus_limit],
    )


def build_candidates(
    expert_actions: list[str],
    expert_action: str,
    *,
    plan: PlanContract,
    max_actions: int,
) -> list[str]:
    pool: list[str] = []
    for action in expert_actions:
        if action not in pool:
            pool.append(action)
    if expert_action not in pool:
        pool.append(expert_action)
    ranked = rank_candidate_actions(
        pool,
        context=plan.rank_context(),
        max_actions=max_actions,
    )
    if expert_action not in ranked:
        ranked = [expert_action] + [action for action in ranked if action != expert_action]
        ranked = ranked[:max_actions]
    return ranked


def main_sample(step: ExpertStep, plan: PlanContract) -> dict[str, Any]:
    state = f"Group action:{step.expert_actions}. Current observation: {step.observation}"
    return {
        "messages": [
            {"role": "system", "content": PLAN_A_MAIN_SYSTEM},
            {
                "role": "user",
                "content": f"Task:\n{step.task}\n\nPlanner state:\n{state}",
            },
            {"role": "assistant", "content": plan.to_tagged_json()},
        ],
        "category": "main",
        "stage": "plan_a_main",
        "source": "multisquare_expert_subtask",
        "schema": "plan_a_v1",
        "source_index": step.source_index,
        "trajectory_step": step.step_index,
        "task_family": step.task_family,
    }


def sub_samples(
    step: ExpertStep,
    plan: PlanContract,
    *,
    max_actions: int,
    include_history: bool,
    history_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action_index, action in enumerate(step.expert_actions):
        observation = step.low_observations[action_index]
        candidates = build_candidates(
            step.expert_actions,
            action,
            plan=plan,
            max_actions=max_actions,
        )
        action_id = expert_action_id(action, candidates)
        if action_id is None:
            continue
        history = (
            recent_history(step, action_index, history_limit) if include_history else None
        )
        messages = build_action_id_messages(
            task=step.task,
            observation=observation,
            candidate_actions=candidates,
            recent_history=history,
            contract=plan.subgoal_block(),
        )
        messages[0]["content"] = PLAN_A_SUB_SYSTEM
        rows.append(
            {
                "messages": messages
                + [
                    {
                        "role": "assistant",
                        "content": format_assistant_action_id(action_id),
                    }
                ],
                "category": "sub",
                "stage": "plan_a_sub_action_id",
                "source": "multisquare_expert_action",
                "schema": "plan_a_v1",
                "source_index": step.source_index,
                "trajectory_step": step.step_index,
                "action_step": action_index,
                "task_family": step.task_family,
                "expert_subgoal": plan.subgoal,
                "expert_action_id": action_id,
                "candidate_actions": candidates,
            }
        )
    return rows


def generate_rows(step: ExpertStep, args: argparse.Namespace) -> list[dict[str, Any]]:
    plan = plan_for_step(step, focus_limit=args.focus_limit)
    rows = [main_sample(step, plan)]
    if not args.main_only:
        rows.extend(
            sub_samples(
                step,
                plan,
                max_actions=args.max_actions,
                include_history=args.include_history,
                history_limit=args.history_limit,
            )
        )
    return rows


def choose_steps(args: argparse.Namespace) -> list[ExpertStep]:
    steps = iter_expert_steps(Path(args.data_dir))
    if args.drop_explicit_target_conflicts:
        steps = [step for step in steps if explicit_target_conflict(step) is None]
    if args.sample_size is None:
        return steps
    if args.sample_size > len(steps):
        raise ValueError(f"--sample-size {args.sample_size} exceeds available steps {len(steps)}")
    rng = random.Random(args.seed)
    return rng.sample(steps, args.sample_size)


def write_split(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = assign_split_for_row(row, args)
        buckets[split].append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in buckets.items():
        write_jsonl(output_dir / f"{split}.jsonl", items)

    main_count = sum(1 for row in rows if row["category"] == "main")
    sub_count = sum(1 for row in rows if row["category"] == "sub")
    manifest = {
        "schema": "plan_a_v1",
        "source": "multisquare_expert_subtask",
        "samples": {split: len(items) for split, items in buckets.items()},
        "categories": {"main": main_count, "sub": sub_count},
        "all_samples": len(rows),
        "focus_limit": args.focus_limit,
        "max_actions": args.max_actions,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def assign_split_for_row(row: dict[str, Any], args: argparse.Namespace) -> str:
    from scienceworld_data import assign_split

    return assign_split(
        f"high_source:{row['source_index']}",
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--output-dir", default="data/plan_a_sft_smoke")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--focus-limit", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--history-limit", type=int, default=4)
    parser.add_argument(
        "--include-history",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--drop-explicit-target-conflicts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = choose_steps(args)
    rows: list[dict[str, Any]] = []
    skipped_sub = 0
    for index, step in enumerate(steps):
        generated = generate_rows(step, args)
        rows.extend(generated)
        if not args.main_only:
            expected_sub = len(step.expert_actions)
            actual_sub = sum(1 for row in generated if row["category"] == "sub")
            skipped_sub += max(expected_sub - actual_sub, 0)
        if (index + 1) % 50 == 0:
            print(f"[plan_a] generated {index + 1}/{len(steps)} expert steps")

    if skipped_sub:
        print(f"[plan_a] skipped {skipped_sub} sub rows without resolvable action-id")
    write_split(rows, args)
    print(f"[plan_a] wrote -> {args.output_dir}")


if __name__ == "__main__":
    main()
