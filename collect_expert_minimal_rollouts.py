"""Collect minimal-contract rollouts by replaying ScienceWorld gold actions.

This produces real environment prompts with valid actions and recent history,
while using the benchmark gold action sequence as the action label source.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from collect_kimi_mas_rollouts import main_user_content, sub_user_content
from eval_episodes import load_episode_list
from generate_minimal_contract_sft_data import (
    CANONICAL_HANDOFF_IF,
    MINIMAL_MAIN_SYSTEM,
    MINIMAL_SUB_SYSTEM,
    MinimalContract,
    extract_target_objects,
    unique_action_guidance,
)
from rollout_schema import ActionStep, MainDecision, SubInvocation, SystemRollout, group_key
from scienceworld_env import EpisodeSpec, ScienceWorldRunner


def choose_episodes(runner: ScienceWorldRunner, args: argparse.Namespace) -> list[EpisodeSpec]:
    if args.episode_list:
        _, specs = load_episode_list(args.episode_list)
        return specs[: args.episodes]
    rng = random.Random(args.seed)
    candidates = []
    task_names = args.tasks or runner.task_names
    for task_name in task_names:
        for variation_id in runner.variations(task_name, args.split):
            candidates.append(EpisodeSpec(task_name, int(variation_id), args.split))
    rng.shuffle(candidates)
    return candidates[: args.episodes]


def build_contract(task: str, observation: str, actions: list[str]) -> MinimalContract:
    action_phrase = "; ".join(actions)
    subgoal = f"Execute the next expert step sequence: {action_phrase}"
    return MinimalContract(
        subgoal=subgoal,
        success_condition="The listed action_guidance sequence has been executed successfully.",
        target_objects=extract_target_objects(subgoal, actions),
        action_guidance=unique_action_guidance(actions, limit=len(actions)),
        handoff_if=CANONICAL_HANDOFF_IF,
    )


def chunk_gold_actions(actions: list[str], chunk_size: int) -> list[list[str]]:
    return [actions[index : index + chunk_size] for index in range(0, len(actions), chunk_size)]


def run_episode(
    runner: ScienceWorldRunner,
    spec: EpisodeSpec,
    args: argparse.Namespace,
    rollout_id: str,
) -> SystemRollout:
    runner.env.load(spec.task_name, spec.variation_id, generateGoldPath=True)
    observation, _ = runner.env.reset()
    observation = str(observation)
    task = str(runner.env.get_task_description())
    gold_actions = runner.gold_actions()
    if gold_actions and gold_actions[0].startswith("ERROR:"):
        raise RuntimeError(gold_actions[0])
    rollout = SystemRollout(
        rollout_id=rollout_id,
        group_key=group_key(spec.task_name, spec.variation_id, spec.split),
        task_name=spec.task_name,
        variation_id=spec.variation_id,
        split=spec.split,
        task_description=task,
        policy_version="scienceworld-gold:minimal",
    )
    done = False
    executed_actions: list[str] = []
    step_count = 0

    for action_chunk in chunk_gold_actions(gold_actions, args.chunk_size):
        if done or step_count >= args.step_limit:
            break
        contract = build_contract(task, observation, action_chunk)
        decision_index = len(rollout.main_decisions)
        decision = MainDecision(
            decision_index=decision_index,
            observation=observation,
            previous_group_actions=list(executed_actions),
            raw_response=contract.to_tagged_json(),
            subtask=contract.subgoal,
            format_valid=True,
            score_before=float(getattr(runner.env, "get_score", lambda: 0.0)()),
            prompt_messages=[
                {"role": "system", "content": MINIMAL_MAIN_SYSTEM},
                {"role": "user", "content": main_user_content(task, observation)},
            ],
        )
        rollout.main_decisions.append(decision)

        invocation = SubInvocation(
            invocation_id=f"sub:{decision_index}",
            parent_main_index=decision_index,
            subtask=contract.to_tagged_json(),
        )
        decision.invocation_id = invocation.invocation_id
        recent_history: list[dict[str, Any]] = []

        for action_index, action in enumerate(action_chunk):
            if done or step_count >= args.step_limit:
                break
            valid_actions = runner.valid_actions()
            prompt_actions = list(valid_actions)
            if args.include_gold_action and action not in set(prompt_actions):
                prompt_actions.append(action)
            sub_messages = [
                {"role": "system", "content": MINIMAL_SUB_SYSTEM},
                {
                    "role": "user",
                    "content": sub_user_content(
                        contract,
                        observation,
                        prompt_actions,
                        recent_history[-args.history_limit :],
                    ),
                },
            ]
            score_before = float(getattr(runner.env, "get_score", lambda: rollout.final_score)())
            next_observation, reward, done, info, action_valid = runner.step(action)
            step_count += 1
            score_after = float(info.get("score", score_before))
            rollout.final_score = score_after
            declared_done = action_index == len(action_chunk) - 1 or done
            handoff = "complete" if declared_done else "continue"
            assistant = (
                f"[action]{action}[/action]"
                f"[subtask_done]{str(declared_done).lower()}[/subtask_done]"
                f"[handoff]{handoff}[/handoff]"
            )
            invocation.steps.append(
                ActionStep(
                    step_index=len(invocation.steps),
                    observation=observation,
                    raw_response=assistant,
                    action=action,
                    format_valid=True,
                    action_valid=action in set(prompt_actions),
                    declared_subtask_done=declared_done,
                    environment_reward=float(reward),
                    score_before=score_before,
                    score_after=score_after,
                    next_observation=next_observation,
                    environment_done=done,
                    prompt_messages=sub_messages,
                    handoff=handoff,
                )
            )
            recent_history.append(
                {
                    "action": action,
                    "format_valid": True,
                    "action_valid": action_valid,
                    "reward": float(reward),
                    "score_before": score_before,
                    "score_after": score_after,
                    "score_delta": score_after - score_before,
                    "handoff": handoff,
                }
            )
            executed_actions.append(action)
            observation = next_observation

        rollout.sub_invocations.append(invocation)

    rollout.environment_done = done
    rollout.truncated = step_count >= args.step_limit and not done
    rollout.validate()
    return rollout


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rollouts: list[SystemRollout]) -> dict[str, float]:
    steps = [step for rollout in rollouts for step in rollout.action_steps]
    return {
        "episodes": len(rollouts),
        "success_rate": sum(rollout.success for rollout in rollouts) / max(len(rollouts), 1),
        "mean_score": sum(rollout.final_score for rollout in rollouts) / max(len(rollouts), 1),
        "mean_steps": len(steps) / max(len(rollouts), 1),
        "main_format_rate": 1.0,
        "sub_format_rate": 1.0,
        "action_valid_rate": sum(step.action_valid for step in steps) / max(len(steps), 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-list", default="")
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--history-limit", type=int, default=6)
    parser.add_argument("--include-gold-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="data/expert_minimal_rollouts/rollouts.jsonl")
    parser.add_argument("--report-output", default="artifacts/expert_minimal_rollouts/report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ScienceWorldRunner(step_limit=args.step_limit)
    rollouts: list[SystemRollout] = []
    errors: list[dict[str, Any]] = []
    try:
        specs = choose_episodes(runner, args)
        for index, spec in enumerate(specs, 1):
            print(f"[expert-minimal] episode {index}/{len(specs)} {spec.task_name} var={spec.variation_id}")
            try:
                rollout = run_episode(runner, spec, args, rollout_id=f"expert_minimal_{index:04d}")
                rollouts.append(rollout)
                print(
                    f"  score={rollout.final_score:.1f} steps={len(rollout.action_steps)} "
                    f"success={rollout.success}"
                )
            except Exception as exc:
                errors.append(
                    {
                        "episode_index": index,
                        "task_name": spec.task_name,
                        "variation_id": spec.variation_id,
                        "split": spec.split,
                        "error": repr(exc),
                    }
                )
                print(f"  failed: {exc}")
    finally:
        runner.close()

    output = Path(args.output)
    write_jsonl(output, [rollout.to_dict() for rollout in rollouts])
    report = {
        "config": vars(args),
        "metrics": summarize(rollouts),
        "errors": errors,
    }
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"[expert-minimal] wrote {output}")
    print(f"[expert-minimal] wrote {report_output}")
    if errors and not rollouts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
