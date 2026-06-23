"""Collect minimal-contract rollouts by replaying ScienceWorld gold actions.

This produces real environment prompts with valid actions and recent history,
while using the benchmark gold action sequence as the action label source.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from collect_kimi_mas_rollouts import main_user_content, sub_user_content
from eval_episodes import generate_stratified_episodes, load_episode_list
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
    if args.k_per_task:
        return generate_stratified_episodes(
            runner,
            args.split,
            args.k_per_task,
            seed=args.seed,
            task_names=args.tasks,
        )
    rng = random.Random(args.seed)
    candidates = []
    task_names = args.tasks or runner.task_names
    for task_name in task_names:
        for variation_id in runner.variations(task_name, args.split):
            candidates.append(EpisodeSpec(task_name, int(variation_id), args.split))
    rng.shuffle(candidates)
    return candidates[: args.episodes]


ACTION_PREFIXES = (
    "open ",
    "close ",
    "go to ",
    "pick up ",
    "look at ",
    "look in ",
    "look on ",
    "examine ",
    "focus on ",
    "move ",
    "pour ",
    "use ",
    "activate ",
    "deactivate ",
    "connect ",
    "disconnect ",
    "read ",
    "mix ",
)
IGNORED_OBJECTS = {"look around", "inventory", "wait1", "wait"}


def clean_object_phrase(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = re.sub(r"\b(in|on|with|from|to) inventory\b", "", text).strip()
    return text.strip(" .")


def action_object(action: str) -> str:
    normalized = clean_object_phrase(action)
    for prefix in ACTION_PREFIXES:
        if normalized.startswith(prefix):
            return clean_object_phrase(normalized[len(prefix) :])
    return normalized


def unique_objects(actions: list[str], limit: int = 6) -> list[str]:
    objects = []
    seen = set()
    for action in actions:
        obj = action_object(action)
        if not obj or obj.isdigit() or obj in IGNORED_OBJECTS:
            continue
        split_done = False
        for splitter in (" to ", " into ", " on ", " in ", " with ", " from "):
            if splitter in obj:
                for part in obj.split(splitter):
                    cleaned = clean_object_phrase(part)
                    if cleaned and cleaned not in seen:
                        seen.add(cleaned)
                        objects.append(cleaned)
                split_done = True
                break
        if not split_done and obj not in seen:
            seen.add(obj)
            objects.append(obj)
        if len(objects) >= limit:
            break
    return objects[:limit]


def objects_for_prefixes(actions: list[str], prefixes: tuple[str, ...], limit: int = 6) -> list[str]:
    objects = []
    seen = set()
    for action in actions:
        normalized = clean_object_phrase(action)
        if not normalized.startswith(prefixes):
            continue
        obj = action_object(action)
        if not obj or obj.isdigit() or obj in IGNORED_OBJECTS:
            continue
        if obj not in seen:
            seen.add(obj)
            objects.append(obj)
        if len(objects) >= limit:
            break
    return objects


def last_room(actions: list[str]) -> str:
    for action in reversed(actions):
        normalized = clean_object_phrase(action)
        for marker in ("go to ", "door to "):
            if marker in normalized:
                return clean_object_phrase(normalized.split(marker)[-1])
    return ""


def semantic_goal(actions: list[str]) -> tuple[str, str]:
    lowered = [clean_object_phrase(action) for action in actions]
    objects = unique_objects(actions)
    object_text = ", ".join(objects[:3]) if objects else "the relevant object"
    room = last_room(actions)
    navigation_actions = sum(
        action.startswith(("go to ", "open door", "close door")) for action in lowered
    )
    manipulation_actions = sum(
        action.startswith(("pick up ", "move ", "pour ", "connect ", "focus on ", "use "))
        for action in lowered
    )

    if room and navigation_actions >= max(1, manipulation_actions):
        return (
            f"Move to the {room}",
            f"The agent is in the {room} or has opened the path to reach it.",
        )
    if any(action.startswith("pick up ") for action in lowered):
        picked = objects_for_prefixes(actions, ("pick up ",)) or objects
        picked_text = ", ".join(picked[:3]) if picked else object_text
        return (
            f"Collect {picked_text}",
            f"The needed object is in the agent inventory: {picked_text}.",
        )
    if any(action.startswith("focus on ") for action in lowered):
        focused = objects_for_prefixes(actions, ("focus on ",)) or objects
        return (
            f"Focus on {focused[0] if focused else 'the target object'}",
            "The target object is focused and the task can check completion.",
        )
    if any(action.startswith(("examine ", "look at ", "look in ", "look on ")) for action in lowered):
        inspected = objects_for_prefixes(actions, ("examine ", "look at ", "look in ", "look on ")) or objects
        inspected_text = ", ".join(inspected[:3]) if inspected else object_text
        return (
            f"Inspect {inspected_text}",
            f"The agent has inspected {inspected_text} and can use the observation for the next decision.",
        )
    if any(action.startswith(("use thermometer", "activate stopwatch", "deactivate stopwatch")) for action in lowered):
        return (
            f"Measure or time {object_text}",
            f"The required measurement for {object_text} has been observed.",
        )
    if any(action.startswith(("connect ", "disconnect ")) for action in lowered):
        return (
            f"Assemble the required circuit connections for {object_text}",
            "The relevant circuit components are connected for the conductivity test.",
        )
    if any(action.startswith(("pour ", "mix ", "move ")) for action in lowered):
        return (
            f"Place or combine {object_text}",
            f"The required objects or substances have been placed or combined: {object_text}.",
        )
    if any(action.startswith(("activate ", "deactivate ", "open ", "close ")) for action in lowered):
        return (
            f"Operate {object_text}",
            f"The relevant device, container, or door has been operated: {object_text}.",
        )
    if any(action.startswith("read ") for action in lowered):
        return (
            f"Read {object_text}",
            f"The information in {object_text} has been read.",
        )
    if any(action.startswith("wait") for action in lowered):
        return (
            "Wait for the current process to advance",
            "The environment has advanced enough for the next observation or life-stage change.",
        )
    return (
        f"Make progress on the current task step involving {object_text}",
        "The current expert-guided step has made observable progress toward the task.",
    )


def build_contract(
    task: str,
    observation: str,
    actions: list[str],
    contract_style: str = "semantic",
) -> MinimalContract:
    action_phrase = "; ".join(actions)
    if contract_style == "action_sequence":
        subgoal = f"Execute the next expert step sequence: {action_phrase}"
        success_condition = "The listed action_guidance sequence has been executed successfully."
        target_objects = extract_target_objects(subgoal, actions)
    else:
        subgoal, success_condition = semantic_goal(actions)
        target_objects = unique_objects(actions) or extract_target_objects(subgoal, actions)
    return MinimalContract(
        subgoal=subgoal,
        success_condition=success_condition,
        target_objects=target_objects,
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
        policy_version=f"scienceworld-gold:minimal:{args.contract_style}",
    )
    done = False
    executed_actions: list[str] = []
    step_count = 0

    for action_chunk in chunk_gold_actions(gold_actions, args.chunk_size):
        if done or step_count >= args.step_limit:
            break
        contract = build_contract(task, observation, action_chunk, args.contract_style)
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
    parser.add_argument("--k-per-task", type=int, default=0)
    parser.add_argument("--episode-list", default="")
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--contract-style", choices=("semantic", "action_sequence"), default="semantic")
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
