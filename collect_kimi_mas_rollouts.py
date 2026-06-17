"""Collect native Kimi Main/Sub rollouts in ScienceWorld.

This is different from contract annotation: Kimi generates the Main contract
and Sub action while interacting with the live environment.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import shutil
import subprocess
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
    r"\[action\](.*?)\[/action\]\s*\[subtask_done\](true|false)\[/subtask_done\]",
    re.DOTALL | re.IGNORECASE,
)
SUB_ACTION_ID_PATTERN = re.compile(
    r"\[action_id\]\s*(A\d+)\s*\[/action_id\]\s*"
    r"\[subtask_done\](true|false)\[/subtask_done\]",
    re.DOTALL | re.IGNORECASE,
)

KIMI_SUB_ACTION_ID_SYSTEM = (
    "You are the ScienceWorld executor in a Main/Sub multi-agent system. "
    "Given a structured contract, current observation, recent execution history, "
    "and a list of candidate action ids, choose exactly one candidate action id "
    "and state whether the current subgoal is complete."
)


def assistant_content_from_stream_json(stdout: str) -> str:
    chunks = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            chunks.append(line)
            continue
        if event.get("role") == "assistant" and event.get("content"):
            chunks.append(str(event["content"]))
    return "\n".join(chunks) if chunks else stdout


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


def parse_sub_response(text: str) -> tuple[str | None, bool, bool]:
    match = SUB_PATTERN.search(text)
    if not match:
        return None, False, False
    return match.group(1).strip(), match.group(2).lower() == "true", True


def parse_sub_action_id_response(text: str) -> tuple[str | None, bool, bool]:
    match = SUB_ACTION_ID_PATTERN.search(text)
    if not match:
        return None, False, False
    return match.group(1).upper(), match.group(2).lower() == "true", True


def snap_action_to_valid(action: str | None, valid_actions: list[str], threshold: float) -> str | None:
    if not action:
        return action
    valid_set = set(valid_actions)
    if action in valid_set:
        return action
    normalized_action = action.lower().strip()
    by_lower = {candidate.lower().strip(): candidate for candidate in valid_actions}
    if normalized_action in by_lower:
        return by_lower[normalized_action]

    source_is_graph_action = normalized_action.startswith("connect ") or normalized_action.startswith("disconnect ")
    candidates = []
    for candidate in valid_actions:
        normalized_candidate = candidate.lower()
        candidate_is_graph_action = normalized_candidate.startswith("connect ") or normalized_candidate.startswith("disconnect ")
        if candidate_is_graph_action and not source_is_graph_action:
            continue
        candidates.append(candidate)
    if not candidates:
        candidates = valid_actions

    best_action = None
    best_score = 0.0
    for candidate in candidates:
        score = difflib.SequenceMatcher(None, normalized_action, candidate.lower()).ratio()
        if score > best_score:
            best_action = candidate
            best_score = score
    if best_action is not None and best_score >= threshold:
        return best_action
    return action


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
    prefix_priority = action_prefix_priority(normalized)
    graph_action = normalized.startswith("connect ") or normalized.startswith("disconnect ")
    priority = prefix_priority
    if graph_action:
        priority += 10
    if normalized.startswith("focus on ") and overlap == 0:
        priority += 8
    return (priority, -overlap, normalized)


def select_candidate_actions(
    actions: list[str],
    *,
    max_actions: int,
    include_graph_actions: bool,
    allow_all_focus_actions: bool = False,
    context: str = "",
) -> list[str]:
    ranked = sorted(actions, key=lambda action: action_rank(action, context))
    if not include_graph_actions:
        non_graph = [
            action
            for action in ranked
            if not action.lower().startswith(("connect ", "disconnect "))
        ]
        if non_graph:
            ranked = non_graph
    if not allow_all_focus_actions:
        focus_filtered = [
            action for action in ranked if not action.lower().startswith("focus on ") or is_safe_focus_action(action)
        ]
        if focus_filtered:
            ranked = focus_filtered
    return ranked[:max_actions]


def is_safe_focus_action(action: str) -> bool:
    normalized = action.lower()
    safe_markers = (
        "substance",
        "water",
        "juice",
        "chocolate",
        "ice",
        "steam",
        "marshmallow",
        "paint",
        "cup containing",
        "pot containing",
        "metal pot",
    )
    unsafe_markers = (
        "stove",
        "oven",
        "freezer",
        "fridge",
        "drawer",
        "door",
        "hallway",
        "kitchen",
        "workshop",
        "cupboard",
        "table",
    )
    return any(marker in normalized for marker in safe_markers) and not any(
        marker in normalized for marker in unsafe_markers
    )


def candidate_action_map(
    actions: list[str],
    *,
    max_actions: int,
    include_graph_actions: bool,
    context: str = "",
) -> list[tuple[str, str]]:
    selected = select_candidate_actions(
        actions,
        max_actions=max_actions,
        include_graph_actions=include_graph_actions,
        allow_all_focus_actions=False,
        context=context,
    )
    return [(f"A{index}", action) for index, action in enumerate(selected)]


def no_progress_actions(recent_history: list[dict[str, Any]]) -> set[str]:
    blocked = set()
    seen = set()
    for item in recent_history:
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        no_score = float(item.get("score_delta", 0.0)) <= 0.0
        no_reward = float(item.get("reward", 0.0)) <= 0.0
        no_observation = not bool(item.get("observation_changed", False))
        repeated = action in seen
        if no_score and no_reward and (no_observation or repeated):
            blocked.add(action)
        seen.add(action)
    return blocked


def remove_blocked_actions(actions: list[str], blocked: set[str]) -> list[str]:
    if not blocked:
        return actions
    filtered = [action for action in actions if action not in blocked]
    return filtered or actions


def action_from_id(action_id: str | None, candidates: list[tuple[str, str]]) -> str | None:
    if action_id is None:
        return None
    by_id = {candidate_id.upper(): action for candidate_id, action in candidates}
    return by_id.get(action_id.upper())


def format_valid_actions(
    actions: list[str],
    max_actions: int,
    max_chars: int,
    context: str = "",
    include_graph_actions: bool = False,
) -> str:
    selected = []
    total = 0
    ranked = select_candidate_actions(
        actions,
        max_actions=max_actions,
        include_graph_actions=include_graph_actions,
        allow_all_focus_actions=False,
        context=context,
    )
    for action in ranked:
        line = f"- {action}"
        if total + len(line) + 1 > max_chars:
            break
        selected.append(line)
        total += len(line) + 1
    suffix = ""
    if len(selected) < len(actions):
        suffix = f"\n... truncated {len(actions) - len(selected)} additional actions"
    return "\n".join(selected) + suffix


class KimiCodeClient:
    def __init__(self, args: argparse.Namespace) -> None:
        api_key = os.environ.get(args.kimicode_api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {args.kimicode_api_key_env}")
        cli_path = args.kimicode_cli_path or shutil.which("kimi")
        if not cli_path:
            default_path = Path.home() / ".kimi-code" / "bin" / "kimi.exe"
            if default_path.exists():
                cli_path = str(default_path)
        if not cli_path:
            raise RuntimeError("Kimi Code CLI not found. Install it or pass --kimicode-cli-path.")
        self.cli_path = cli_path
        self.timeout = args.kimicode_timeout
        self.env = os.environ.copy()
        self.env.update(
            {
                "KIMI_MODEL_NAME": args.kimicode_model,
                "KIMI_MODEL_API_KEY": api_key,
                "KIMI_MODEL_PROVIDER_TYPE": args.kimicode_provider_type,
                "KIMI_MODEL_BASE_URL": args.kimicode_base_url,
                "KIMI_MODEL_TEMPERATURE": str(args.temperature),
                "KIMI_DISABLE_TELEMETRY": "1",
            }
        )

    def prompt(self, text: str) -> str:
        try:
            completed = subprocess.run(
                [self.cli_path, "-p", text, "--output-format", "stream-json"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env=self.env,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Kimi Code CLI timed out after {self.timeout:.1f}s") from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"Kimi Code CLI failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        return assistant_content_from_stream_json(completed.stdout)


def main_prompt(task: str, observation: str, previous_actions: list[str]) -> str:
    return (
        f"{CONTRACT_MAIN_SYSTEM}\n"
        f"{DISTILL_SYSTEM}\n"
        "ScienceWorld planning rules:\n"
        "- Choose exactly one short subgoal achievable in 1-4 environment actions.\n"
        "- Do not plan the whole task at once.\n"
        "- The subgoal success_condition must be directly checkable from the next observations.\n"
        "- If the agent needs to move, make the subgoal only about reaching the next room.\n"
        "- If the agent needs an object, make the subgoal only about finding or taking that object.\n"
        "Return only the contract block. Do not use markdown or explain outside the block.\n\n"
        f"Task:\n{task}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"Actions completed since previous plan:\n{json.dumps(previous_actions, ensure_ascii=False)}"
    )


def main_user_content(task: str, observation: str, previous_actions: list[str]) -> str:
    return (
        f"Task:\n{task}\n\nCurrent observation:\n{observation}\n\n"
        f"Actions completed since previous plan:\n{previous_actions}"
    )


def sub_user_content(
    contract: CommunicationContract,
    observation: str,
    candidate_actions: list[tuple[str, str]],
    recent_history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    action_lines = [f"{action_id}: {action}" for action_id, action in candidate_actions]
    valid_text = "\n".join(action_lines)
    history_text = json.dumps(recent_history[-args.history_limit :], ensure_ascii=False, indent=2)
    return (
        f"Contract:\n{contract.to_tagged_json()}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Recent execution history:\n{history_text}\n\n"
        f"Candidate actions:\n{valid_text}"
    )


def sub_prompt(
    contract: CommunicationContract,
    observation: str,
    candidate_actions: list[tuple[str, str]],
    recent_history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    if not args.use_action_ids:
        action_text = "\n".join(action for _, action in candidate_actions)
        return (
            f"{CONTRACT_SUB_SYSTEM}\n"
            "Return only [action]...[/action][subtask_done]true|false[/subtask_done]. "
            "The action must be copied exactly from the valid actions list when possible.\n\n"
            f"Contract:\n{contract.to_tagged_json()}\n\n"
            f"Observation:\n{observation}\n\n"
            f"Recent execution history:\n{json.dumps(recent_history[-args.history_limit:], ensure_ascii=False, indent=2)}\n\n"
            f"Valid actions:\n{action_text}"
        )
    return (
        f"{KIMI_SUB_ACTION_ID_SYSTEM}\n"
        "ScienceWorld execution rules:\n"
        "- Choose exactly one candidate action id from the Candidate actions list.\n"
        "- Return only [action_id]A<num>[/action_id][subtask_done]true|false[/subtask_done].\n"
        "- Do not repeat an action if recent history shows it gave no reward, no score gain, and no useful observation change.\n"
        "- Prefer actions that change room, open containers/doors, pick up needed objects, activate/deactivate devices, examine/focus target substances, or advance the success_condition.\n"
        "- Use subtask_done=true only when the contract success_condition is already satisfied by the current observation or the chosen action will satisfy it.\n"
        "- Never invent an action string.\n\n"
        f"{sub_user_content(contract, observation, candidate_actions, recent_history, args)}"
    )


def repair_prompt(raw: str, observation: str, valid_actions: list[str], args: argparse.Namespace) -> str:
    valid_text = format_valid_actions(
        valid_actions,
        args.max_valid_actions,
        args.max_valid_action_chars,
        observation,
        include_graph_actions=args.include_graph_actions,
    )
    return (
        "Repair the ScienceWorld executor response. Return only "
        "[action]...[/action][subtask_done]true|false[/subtask_done]. "
        "Choose one action copied exactly from the valid actions list.\n\n"
        f"Previous response:\n{raw}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Valid actions:\n{valid_text}"
    )


def repair_action_id_prompt(
    raw: str,
    observation: str,
    candidate_actions: list[tuple[str, str]],
    recent_history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    action_text = "\n".join(f"{action_id}: {action}" for action_id, action in candidate_actions)
    history_text = json.dumps(recent_history[-args.history_limit :], ensure_ascii=False, indent=2)
    return (
        "Repair the ScienceWorld executor response. Return only "
        "[action_id]A<num>[/action_id][subtask_done]true|false[/subtask_done]. "
        "Choose one id exactly from Candidate actions. Do not explain.\n\n"
        f"Previous response:\n{raw}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Recent execution history:\n{history_text}\n\n"
        f"Candidate actions:\n{action_text}"
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
    agent: KimiCodeClient,
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
        policy_version=f"kimicode-native:{args.kimicode_model}",
    )
    previous_actions: list[str] = []
    step_count = 0
    done = False

    while not done and step_count < args.step_limit and len(rollout.main_decisions) < args.max_subtasks:
        main_messages = [
            {"role": "system", "content": CONTRACT_MAIN_SYSTEM},
            {"role": "user", "content": main_user_content(task, observation, previous_actions)},
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
        subtask_done = False
        previous_actions = []
        recent_history: list[dict[str, Any]] = []

        while not done and not subtask_done and step_count < args.step_limit:
            valid_actions = runner.valid_actions()
            candidate_source_actions = remove_blocked_actions(
                valid_actions,
                no_progress_actions(recent_history) if args.block_no_progress_repeats else set(),
            )
            context = contract.to_tagged_json() + "\n" + observation
            candidate_actions = candidate_action_map(
                candidate_source_actions,
                max_actions=args.max_valid_actions,
                include_graph_actions=args.include_graph_actions,
                context=context,
            )
            sub_user = sub_user_content(contract, observation, candidate_actions, recent_history, args)
            sub_messages = [
                {"role": "system", "content": KIMI_SUB_ACTION_ID_SYSTEM if args.use_action_ids else CONTRACT_SUB_SYSTEM},
                {"role": "user", "content": sub_user},
            ]
            sub_raw = agent.prompt(sub_prompt(contract, observation, candidate_actions, recent_history, args))
            if args.use_action_ids:
                action_id, declared_done, format_valid = parse_sub_action_id_response(sub_raw)
                action = action_from_id(action_id, candidate_actions)
            else:
                action, declared_done, format_valid = parse_sub_response(sub_raw)

            action_valid_precheck = action in set(valid_actions) if action else False
            if (
                args.snap_invalid_actions
                and not args.use_action_ids
                and format_valid
                and action is not None
                and not action_valid_precheck
            ):
                snapped = snap_action_to_valid(action, valid_actions, args.snap_threshold)
                if snapped != action:
                    action = snapped
                    action_valid_precheck = action in set(valid_actions)

            if args.repair_invalid_actions and (not format_valid or not action_valid_precheck):
                if args.use_action_ids:
                    repair_raw = agent.prompt(
                        repair_action_id_prompt(sub_raw, observation, candidate_actions, recent_history, args)
                    )
                    repaired_action_id, repaired_done, repaired_format = parse_sub_action_id_response(repair_raw)
                    repaired_action = action_from_id(repaired_action_id, candidate_actions)
                else:
                    repair_raw = agent.prompt(repair_prompt(sub_raw, observation, valid_actions, args))
                    repaired_action, repaired_done, repaired_format = parse_sub_response(repair_raw)
                if repaired_format:
                    sub_raw = repair_raw
                    action, declared_done, format_valid = repaired_action, repaired_done, repaired_format

            try:
                score_before = float(runner.env.get_score())
            except Exception:
                score_before = rollout.final_score

            if action is None:
                next_observation = observation
                reward = 0.0
                info: dict[str, Any] = {"score": score_before, "format_error": True}
                action_valid = False
            else:
                next_observation, reward, done, info, action_valid = runner.step(action)
                step_count += 1

            score_after = float(info.get("score", score_before))
            rollout.final_score = score_after
            observation_changed = next_observation.strip() != observation.strip()
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
                    "observation_changed": observation_changed,
                }
            )
            observation = next_observation
            subtask_done = declared_done

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
    parser.add_argument("--max-valid-actions", type=int, default=200)
    parser.add_argument("--max-valid-action-chars", type=int, default=12000)
    parser.add_argument("--include-graph-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-action-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history-limit", type=int, default=6)
    parser.add_argument("--block-no-progress-repeats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-invalid-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--snap-invalid-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--snap-threshold", type=float, default=0.72)
    parser.add_argument("--kimicode-model", default="kimi-for-coding")
    parser.add_argument("--kimicode-base-url", default="https://api.kimi.com/coding/v1")
    parser.add_argument("--kimicode-provider-type", default="kimi")
    parser.add_argument("--kimicode-api-key-env", default="KIMI_CODE_API_KEY")
    parser.add_argument("--kimicode-cli-path", default="")
    parser.add_argument("--kimicode-timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="data/kimi_mas_rollouts/rollouts.jsonl")
    parser.add_argument("--report-output", default="artifacts/kimi_mas_rollouts/report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = KimiCodeClient(args)
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
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"kimicode_cli_path"}
        },
        "metrics": summarize(rollouts),
        "errors": errors,
    }
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"[kimi-mas] wrote {output}")
    print(f"[kimi-mas] wrote {report_output}")


if __name__ == "__main__":
    main()
