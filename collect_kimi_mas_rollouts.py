"""Collect native Kimi Main/Sub rollouts in ScienceWorld.

Kimi generates the Main contract and Sub action while interacting with
the live ScienceWorld environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from contract_schema import CommunicationContract, parse_contract_text
from eval_episodes import load_episode_list
from generate_contract_sft_data import (
    CONTRACT_MAIN_SYSTEM,
    CONTRACT_SUB_SYSTEM,
    DISTILL_SYSTEM,
    extract_first_json_object,
)
from rollout_schema import (
    ActionStep,
    MainDecision,
    SubInvocation,
    SystemRollout,
    group_key,
)
from scienceworld_env import EpisodeSpec, ScienceWorldRunner


SUB_PATTERN = re.compile(
    r"\[action\](.*?)\[/action\]\s*"
    r"\[subtask_done\](true|false)\[/subtask_done\]\s*"
    r"(?:\[handoff\](continue|complete|blocked|need_replan)\[/handoff\])?",
    re.DOTALL | re.IGNORECASE,
)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def parse_contract_response(text: str) -> CommunicationContract | None:
    candidates = [strip_code_fence(text)]
    unescaped = text.replace('\\"', '"').replace("\\n", "\n")
    if unescaped != text:
        candidates.append(strip_code_fence(unescaped))
    for candidate in candidates:
        try:
            return parse_contract_text(candidate)
        except Exception:
            try:
                return parse_contract_text(extract_first_json_object(candidate))
            except Exception:
                pass
    return None


def parse_sub_response(text: str) -> tuple[str | None, bool, str, bool]:
    match = SUB_PATTERN.search(text)
    if not match:
        return None, False, "continue", False
    done = match.group(2).lower() == "true"
    handoff = (match.group(3) or ("complete" if done else "continue")).lower()
    return match.group(1).strip(), done, handoff, True


ACTION_PREFIX_PRIORITIES = (
    ("look", 0),
    ("inventory", 0),
    ("open", 1),
    ("go", 1),
    ("pick up", 1),
    ("close", 2),
    ("examine", 2),
    ("focus", 2),
    ("activate", 2),
    ("deactivate", 2),
    ("use", 3),
    ("move", 4),
    ("pour", 4),
    ("put", 4),
    ("wait", 5),
)


def action_prefix_priority(action: str) -> int:
    if action in {"look around", "inventory"}:
        return 0
    if action.startswith("look at ") or action.startswith("look in "):
        return 3
    for prefix, priority in ACTION_PREFIX_PRIORITIES:
        if action == prefix or action.startswith(prefix + " "):
            return priority
    return 6


def action_rank(action: str, context: str = "") -> tuple[int, int, str]:
    normalized = action.lower()
    context_tokens = {
        token
        for token in re.findall(r"[a-z0-9_]+", context.lower())
        if len(token) >= 3
    }
    action_tokens = set(re.findall(r"[a-z0-9_]+", normalized))
    overlap = len(context_tokens & action_tokens)
    priority = action_prefix_priority(normalized)
    return (priority, -overlap, normalized)


def select_candidate_actions(
    actions: list[str],
    *,
    max_actions: int = 0,
    rank_actions: bool = False,
    context: str = "",
) -> list[str]:
    ranked = sorted(actions, key=lambda action: action_rank(action, context)) if rank_actions else list(actions)
    if max_actions > 0:
        ranked = ranked[:max_actions]
    return ranked


class KimiHttpClient:
    def __init__(self, args: argparse.Namespace) -> None:
        api_key = os.environ.get(args.api_key_env)
        if not api_key and args.api_key_file:
            api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {args.api_key_env} or pass --api-key-file."
            )
        self.api_key = api_key
        self.api_base = args.api_base.rstrip("/")
        self.model = args.model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.request_timeout = args.request_timeout

    def _post_json(self, endpoint: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Kimi API failed with HTTP {exc.code}: {body}") from exc

    def prompt(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = self._post_json(
            f"{self.api_base}/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            payload,
        )
        chunks = data.get("content", [])
        return "\n".join(
            str(chunk.get("text", ""))
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("type") == "text"
        )


def main_prompt(task: str, observation: str, previous_actions: list[str]) -> str:
    return (
        f"{CONTRACT_MAIN_SYSTEM}\n"
        f"{DISTILL_SYSTEM}\n"
        "ScienceWorld planning rules:\n"
        "- Choose exactly one short subgoal achievable in 1-4 environment actions.\n"
        "- Do not plan the whole task at once.\n"
        "- The subgoal success_condition must be directly checkable from the next observations.\n"
        "- Include action_guidance that names likely executable actions.\n"
        "- If the agent needs to move, make the subgoal only about reaching the next room.\n"
        "- If the agent needs an object, make the subgoal only about finding or taking that object.\n"
        "Return only the contract block. Do not use markdown or explain outside the block.\n\n"
        f"Task:\n{task}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"Actions completed since previous plan:\n{json.dumps(previous_actions, ensure_ascii=False)}"
    )


def main_user_content(task: str, observation: str) -> str:
    return f"Task:\n{task}\n\nPlanner state:\n{observation}"


def sub_user_content(
    contract: CommunicationContract,
    observation: str,
    valid_actions: list[str],
    recent_history: list[dict[str, Any]],
) -> str:
    valid_text = "\n".join(f"- {action}" for action in valid_actions)
    history_text = json.dumps(recent_history, ensure_ascii=False, indent=2)
    return (
        f"Contract:\n{contract.to_tagged_json()}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Recent execution history:\n{history_text}\n\n"
        f"Valid actions:\n{valid_text}"
    )


def sub_prompt(
    contract: CommunicationContract,
    observation: str,
    valid_actions: list[str],
    recent_history: list[dict[str, Any]],
) -> str:
    return (
        f"{CONTRACT_SUB_SYSTEM}\n"
        "Return only [action]...[/action][subtask_done]true|false[/subtask_done]"
        "[handoff]continue|complete|blocked|need_replan[/handoff]. "
        "Copy the action exactly from the valid actions list. Set subtask_done=true "
        "only when the contract success condition is satisfied. Use handoff=complete "
        "when the contract is done, blocked when no listed action can make progress, "
        "and need_replan when the contract no longer matches the current state.\n\n"
        f"{sub_user_content(contract, observation, valid_actions, recent_history)}"
    )


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


def run_episode(
    agent: KimiHttpClient,
    runner: ScienceWorldRunner,
    spec: EpisodeSpec,
    args: argparse.Namespace,
    rollout_id: str,
) -> SystemRollout:
    observation, task, _ = runner.reset(spec)
    rollout = SystemRollout(
        rollout_id=rollout_id,
        group_key=group_key(spec.task_name, spec.variation_id, spec.split),
        task_name=spec.task_name,
        variation_id=spec.variation_id,
        split=spec.split,
        task_description=task,
        policy_version=f"kimi-native:{args.model}",
    )
    previous_actions: list[str] = []
    step_count = 0
    done = False

    while not done and step_count < args.step_limit and len(rollout.main_decisions) < args.max_subtasks:
        main_messages = [
            {"role": "system", "content": CONTRACT_MAIN_SYSTEM},
            {"role": "user", "content": main_user_content(task, observation)},
        ]
        main_raw = agent.prompt(main_prompt(task, observation, previous_actions))
        contract = parse_contract_response(main_raw)
        decision = MainDecision(
            decision_index=len(rollout.main_decisions),
            observation=observation,
            previous_group_actions=list(previous_actions),
            raw_response=main_raw,
            subtask=contract.subgoal if contract else None,
            format_valid=contract is not None,
            score_before=float(getattr(runner.env, "get_score", lambda: 0.0)()),
            prompt_messages=main_messages,
        )
        rollout.main_decisions.append(decision)
        if contract is None:
            break

        invocation = SubInvocation(
            invocation_id=f"sub:{decision.decision_index}",
            parent_main_index=decision.decision_index,
            subtask=contract.to_tagged_json(),
        )
        decision.invocation_id = invocation.invocation_id
        declared_done = False
        previous_actions = []
        recent_history: list[dict[str, Any]] = []

        while (
            not done
            and not declared_done
            and step_count < args.step_limit
            and len(invocation.steps) < args.max_steps_per_subtask
        ):
            valid_actions = runner.valid_actions()
            context = contract.to_tagged_json() + "\n" + observation
            ranked_actions = select_candidate_actions(
                valid_actions,
                max_actions=args.max_valid_actions,
                rank_actions=args.rank_valid_actions,
                context=context,
            )
            sub_user = sub_user_content(
                contract,
                observation,
                ranked_actions,
                recent_history[-args.history_limit :],
            )
            sub_messages = [
                {
                    "role": "system",
                    "content": (
                        CONTRACT_SUB_SYSTEM
                        + " Include [handoff]continue|complete|blocked|need_replan[/handoff]."
                    ),
                },
                {"role": "user", "content": sub_user},
            ]
            sub_raw = agent.prompt(
                sub_prompt(
                    contract,
                    observation,
                    ranked_actions,
                    recent_history[-args.history_limit :],
                )
            )
            action, declared_done, handoff, format_valid = parse_sub_response(sub_raw)
            action_valid = action in set(valid_actions) if action else False

            try:
                score_before = float(runner.env.get_score())
            except Exception:
                score_before = rollout.final_score

            if action is None:
                next_observation = observation
                reward = 0.0
            else:
                next_observation, reward, done, info, action_valid = runner.step(action)
                step_count += 1

            score_after = float(info.get("score", score_before)) if action is not None else score_before
            rollout.final_score = score_after
            invocation.steps.append(
                ActionStep(
                    step_index=len(invocation.steps),
                    observation=observation,
                    raw_response=sub_raw,
                    action=action,
                    format_valid=format_valid,
                    action_valid=action_valid,
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
            if action is None:
                break
            previous_actions.append(action)
            recent_history.append(
                {
                    "action": action,
                    "format_valid": format_valid,
                    "action_valid": action_valid,
                    "reward": float(reward),
                    "score_before": score_before,
                    "score_after": score_after,
                    "score_delta": score_after - score_before,
                    "handoff": handoff,
                }
            )
            observation = next_observation
            declared_done = declared_done or handoff in {"complete", "blocked", "need_replan"}

        rollout.sub_invocations.append(invocation)

    rollout.environment_done = done
    rollout.truncated = step_count >= args.step_limit
    rollout.validate()
    return rollout


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rollouts: list[SystemRollout]) -> dict[str, float]:
    steps = [step for rollout in rollouts for step in rollout.action_steps]
    return {
        "episodes": len(rollouts),
        "success_rate": sum(rollout.success for rollout in rollouts) / max(len(rollouts), 1),
        "mean_score": sum(rollout.final_score for rollout in rollouts) / max(len(rollouts), 1),
        "mean_steps": len(steps) / max(len(rollouts), 1),
        "main_format_rate": (
            sum(decision.format_valid for rollout in rollouts for decision in rollout.main_decisions)
            / max(sum(len(rollout.main_decisions) for rollout in rollouts), 1)
        ),
        "sub_format_rate": sum(step.format_valid for step in steps) / max(len(steps), 1),
        "action_valid_rate": sum(step.action_valid for step in steps) / max(len(steps), 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-list", default="")
    parser.add_argument("--step-limit", type=int, default=30)
    parser.add_argument("--max-subtasks", type=int, default=10)
    parser.add_argument("--max-steps-per-subtask", type=int, default=6)
    parser.add_argument("--history-limit", type=int, default=6)
    parser.add_argument("--max-valid-actions", type=int, default=0, help="0 means include every environment valid action")
    parser.add_argument("--rank-valid-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default="kimi-for-coding")
    parser.add_argument("--api-base", default="https://api.kimi.com/coding")
    parser.add_argument("--api-key-env", default="KIMI_CODE_API_KEY")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="data/kimi_mas_rollouts/rollouts.jsonl")
    parser.add_argument("--report-output", default="artifacts/kimi_mas_rollouts/report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = KimiHttpClient(args)
    runner = ScienceWorldRunner(step_limit=args.step_limit)
    rollouts: list[SystemRollout] = []
    errors: list[dict[str, Any]] = []
    output = Path(args.output)
    if output.exists():
        output.unlink()
    try:
        specs = choose_episodes(runner, args)
        for index, spec in enumerate(specs, 1):
            print(f"[kimi-mas] episode {index}/{len(specs)} {spec.task_name} var={spec.variation_id}")
            try:
                rollout = run_episode(agent, runner, spec, args, rollout_id=f"kimi_native_{index:04d}")
                rollouts.append(rollout)
                append_jsonl(output, rollout.to_dict())
                print(
                    f"  score={rollout.final_score:.1f} steps={len(rollout.action_steps)} "
                    f"success={rollout.success}"
                )
            except Exception as exc:
                error = {
                    "episode_index": index,
                    "task_name": spec.task_name,
                    "variation_id": spec.variation_id,
                    "split": spec.split,
                    "error": repr(exc),
                }
                errors.append(error)
                print(f"  failed: {exc}")
    finally:
        runner.close()

    report = {
        "config": vars(args),
        "metrics": summarize(rollouts),
        "errors": errors,
    }
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"[kimi-mas] wrote {output}")
    print(f"[kimi-mas] wrote {report_output}")
    if errors and not rollouts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
