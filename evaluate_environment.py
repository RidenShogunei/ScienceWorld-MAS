"""Run hierarchical Main/Sub adapters inside the official ScienceWorld environment."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval_episodes import (
    episode_list_metadata,
    generate_stratified_episodes,
    load_episode_list,
    save_episode_list,
)
from eval_metrics import attach_multi_square_episode_fields, multi_square_aggregate
from collect_kimi_mas_rollouts import parse_minimal_contract_response, parse_sub_response
from generate_sft_data import MAIN_SYSTEM, SUB_SYSTEM
from generate_minimal_contract_sft_data import MINIMAL_MAIN_SYSTEM, MINIMAL_SUB_SYSTEM
from l1.protocol import build_action_id_messages, decode_action_id, parse_action_id_response, rank_candidate_actions
from plan_a.schema import PLAN_A_MAIN_SYSTEM, PLAN_A_SUB_SYSTEM, parse_plan_response
from provenance import experiment_provenance
from scienceworld_env import EpisodeSpec, ScienceWorldRunner
from sft_trainer import ensure_torch_set_submodule


MAIN_PATTERN = re.compile(r"\[subtask\](.*?)\[/subtask\]", re.DOTALL)
SUB_PATTERN = re.compile(
    r"\[action\](.*?)\[/action\]\s*\[subtask_done\](true|false)\[/subtask_done\]",
    re.DOTALL | re.IGNORECASE,
)


def main_messages(
    task: str,
    observation: str,
    group_actions: list[str],
    agent_interface: str,
) -> list[dict[str, str]]:
    state = f"Group action:{group_actions}. Current observation: {observation}"
    if agent_interface == "contract-simple":
        return [
            {"role": "system", "content": MINIMAL_MAIN_SYSTEM},
            {"role": "user", "content": f"Task:\n{task}\n\nPlanner state:\n{state}"},
        ]
    return [
        {"role": "system", "content": MAIN_SYSTEM},
        {"role": "user", "content": f"Task:\n{task}\n\nPlanner state:\n{state}"},
    ]


def plan_a_main_messages(
    task: str,
    observation: str,
    group_actions: list[str],
) -> list[dict[str, str]]:
    state = f"Group action:{group_actions}. Current observation: {observation}"
    return [
        {"role": "system", "content": PLAN_A_MAIN_SYSTEM},
        {"role": "user", "content": f"Task:\n{task}\n\nPlanner state:\n{state}"},
    ]


def sub_messages(
    plan: str,
    observation: str,
    agent_interface: str,
) -> list[dict[str, str]]:
    if agent_interface == "contract-simple":
        return [
            {"role": "system", "content": MINIMAL_SUB_SYSTEM},
            {
                "role": "user",
                "content": f"Contract:\n{plan}\n\nObservation:\n{observation}",
            },
        ]
    return [
        {"role": "system", "content": SUB_SYSTEM},
        {
            "role": "user",
            "content": f"Subtask:\n{plan}\n\nObservation:\n{observation}",
        },
    ]


class HierarchicalPolicy:
    def __init__(self, base_model: str, main_adapter: str, sub_adapter: str, use_4bit: bool) -> None:
        ensure_torch_set_submodule()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if use_4bit:
            kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map="auto",
            )
        elif torch.cuda.is_available():
            kwargs.update(dtype=torch.bfloat16, device_map={"": 0})
        base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        self.model = PeftModel.from_pretrained(base, main_adapter, adapter_name="main")
        self.model.load_adapter(sub_adapter, adapter_name="sub")
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, adapter: str, messages: list[dict], max_input_length: int, max_new_tokens: int) -> str:
        self.model.set_adapter(adapter)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    def plan(self, task: str, observation: str, group_actions: list[str], args) -> tuple[str | None, str]:
        if args.agent_interface == "plan-a":
            text = self.generate(
                "main",
                plan_a_main_messages(task, observation, group_actions),
                args.max_input_length,
                args.main_max_new_tokens,
            )
            plan_contract = parse_plan_response(text)
            return (plan_contract.subgoal_block() if plan_contract else None), text
        interface = "contract-simple" if args.agent_interface == "action-id" else args.agent_interface
        text = self.generate(
            "main",
            main_messages(task, observation, group_actions, interface),
            args.max_input_length,
            args.main_max_new_tokens,
        )
        if args.agent_interface in {"contract-simple", "action-id"}:
            contract = parse_minimal_contract_response(text)
            return (contract.to_tagged_json() if contract else None), text
        match = MAIN_PATTERN.search(text)
        return (match.group(1).strip() if match else None), text

    def act(self, plan: str, observation: str, args) -> tuple[str | None, bool, str, str]:
        text = self.generate(
            "sub",
            sub_messages(plan, observation, args.agent_interface),
            args.max_input_length,
            args.sub_max_new_tokens,
        )
        if args.agent_interface == "contract-simple":
            action, done, handoff, valid = parse_sub_response(text)
            return (action if valid else None), done, handoff, text
        match = SUB_PATTERN.search(text)
        if not match:
            return None, False, "continue", text
        done = match.group(2).lower() == "true"
        return match.group(1).strip(), done, ("complete" if done else "continue"), text

    def act_action_id(
        self,
        task: str,
        contract: str,
        observation: str,
        candidate_actions: list[str],
        recent_history: list[dict],
        args,
    ) -> tuple[str | None, str, bool]:
        messages = build_action_id_messages(
            task=task,
            observation=observation,
            candidate_actions=candidate_actions,
            recent_history=recent_history[-args.history_limit :],
            contract=contract,
        )
        if args.agent_interface == "plan-a":
            messages[0]["content"] = PLAN_A_SUB_SYSTEM
        text = self.generate(
            "sub",
            messages,
            args.max_input_length,
            args.sub_max_new_tokens,
        )
        action_id = parse_action_id_response(text)
        action = decode_action_id(action_id, candidate_actions)
        return action, text, action_id is not None


def run_episode_action_id(
    policy: HierarchicalPolicy,
    runner: ScienceWorldRunner,
    spec: EpisodeSpec,
    args,
) -> dict:
    """Per-step Main plan/contract + Sub action-id (L1 minimal or Plan A)."""
    observation, task, reset_info = runner.reset(spec)
    steps_log: list[dict] = []
    total_reward = 0.0
    invalid_actions = 0
    main_format_errors = 0
    sub_parse_errors = 0
    step_count = 0
    done = False
    episode_actions: list[str] = []
    recent_history: list[dict] = []

    while not done and step_count < args.step_limit:
        plan, main_raw = policy.plan(task, observation, episode_actions, args)
        if plan is None:
            main_format_errors += 1
            steps_log.append(
                {
                    "observation": observation,
                    "main_raw": main_raw,
                    "main_format_valid": False,
                }
            )
            break

        valid_actions = runner.valid_actions()
        if args.agent_interface == "plan-a":
            plan_contract = parse_plan_response(main_raw)
            rank_context = plan_contract.rank_context() if plan_contract else f"{plan}\n{observation}"
        else:
            rank_context = f"{plan}\n{observation}"
        candidates = rank_candidate_actions(
            valid_actions,
            context=rank_context,
            max_actions=args.max_candidate_actions,
        )
        action, sub_raw, parse_ok = policy.act_action_id(
            task,
            plan,
            observation,
            candidates,
            recent_history,
            args,
        )
        if not parse_ok or action is None:
            sub_parse_errors += 1
            steps_log.append(
                {
                    "observation": observation,
                    "main_raw": main_raw,
                    "contract": plan,
                    "sub_raw": sub_raw,
                    "main_format_valid": True,
                    "sub_parse_valid": False,
                    "num_candidates": len(candidates),
                }
            )
            break

        next_observation, reward, done, info, action_valid = runner.step(action)
        step_count += 1
        total_reward += reward
        invalid_actions += int(not action_valid)
        step_score = float(info.get("score", 0.0))
        print(
            f"  step {step_count}/{args.step_limit} score={step_score:.1f} done={done}",
            flush=True,
        )
        step_record = {
            "observation": observation,
            "main_raw": main_raw,
            "contract": plan,
            "sub_raw": sub_raw,
            "action": action,
            "main_format_valid": True,
            "sub_parse_valid": True,
            "action_valid": action_valid,
            "num_candidates": len(candidates),
            "reward": reward,
            "score": float(info.get("score", 0.0)),
            "next_observation": next_observation,
            "environment_done": done,
        }
        steps_log.append(step_record)
        recent_history.append(
            {"action": action, "action_valid": action_valid, "subtask_done": False}
        )
        episode_actions.append(action)
        observation = next_observation

    final_score = steps_log[-1].get("score", 0.0) if steps_log and "score" in steps_log[-1] else 0.0
    format_errors = main_format_errors + sub_parse_errors
    return attach_multi_square_episode_fields(
        {
            "task_name": spec.task_name,
            "variation_id": spec.variation_id,
            "split": spec.split,
            "task_description": task,
            "reset_info": reset_info,
            "groups": [{"contract": None, "main_raw": None, "steps": steps_log}],
            "steps": step_count,
            "total_reward": total_reward,
            "final_score": final_score,
            "success": final_score >= 100.0,
            "invalid_actions": invalid_actions,
            "action_valid_rate": (step_count - invalid_actions) / max(step_count, 1),
            "format_errors": format_errors,
            "main_format_errors": main_format_errors,
            "sub_parse_errors": sub_parse_errors,
            "environment_done": done,
        }
    )


def choose_episodes(runner: ScienceWorldRunner, args) -> list[EpisodeSpec]:
    rng = random.Random(args.seed)
    task_names = args.tasks or runner.task_names
    candidates = []
    for task_name in task_names:
        variations = runner.variations(task_name, args.split)
        for variation_id in variations:
            candidates.append(EpisodeSpec(task_name, int(variation_id), args.split))
    rng.shuffle(candidates)
    return candidates[: args.episodes]


def run_episode(policy: HierarchicalPolicy, runner: ScienceWorldRunner, spec: EpisodeSpec, args) -> dict:
    if args.agent_interface in {"action-id", "plan-a"}:
        return run_episode_action_id(policy, runner, spec, args)

    observation, task, reset_info = runner.reset(spec)
    groups = []
    total_reward = 0.0
    invalid_actions = 0
    format_errors = 0
    step_count = 0
    done = False
    previous_group_actions: list[str] = []

    while not done and step_count < args.step_limit and len(groups) < args.max_subtasks:
        plan, main_raw = policy.plan(task, observation, previous_group_actions, args)
        group = {
            "subtask": plan if args.agent_interface == "legacy" else None,
            "contract": plan if args.agent_interface == "contract-simple" else None,
            "main_raw": main_raw,
            "steps": [],
        }
        groups.append(group)
        if plan is None:
            format_errors += 1
            break

        current_group_actions = []
        subtask_done = False
        while not done and not subtask_done and step_count < args.step_limit:
            action, subtask_done, handoff, sub_raw = policy.act(plan, observation, args)
            if action is None:
                format_errors += 1
                group["steps"].append(
                    {"observation": observation, "sub_raw": sub_raw, "format_valid": False}
                )
                break
            next_observation, reward, done, info, action_valid = runner.step(action)
            step_count += 1
            total_reward += reward
            invalid_actions += int(not action_valid)
            group["steps"].append(
                {
                    "observation": observation,
                    "sub_raw": sub_raw,
                    "action": action,
                    "subtask_done": subtask_done,
                    "handoff": handoff,
                    "action_valid": action_valid,
                    "reward": reward,
                    "score": float(info.get("score", 0.0)),
                    "next_observation": next_observation,
                    "environment_done": done,
                }
            )
            current_group_actions.append(action)
            observation = next_observation
            if handoff in {"blocked", "need_replan"}:
                subtask_done = True
        previous_group_actions = current_group_actions

    final_score = 0.0
    if groups and groups[-1]["steps"]:
        final_score = groups[-1]["steps"][-1].get("score", 0.0)
    return attach_multi_square_episode_fields(
        {
            "task_name": spec.task_name,
            "variation_id": spec.variation_id,
            "split": spec.split,
            "task_description": task,
            "reset_info": reset_info,
            "groups": groups,
            "steps": step_count,
            "total_reward": total_reward,
            "final_score": final_score,
            "success": final_score >= 100.0,
            "invalid_actions": invalid_actions,
            "action_valid_rate": (step_count - invalid_actions) / max(step_count, 1),
            "format_errors": format_errors,
            "environment_done": done,
        }
    )


def resolve_episode_specs(runner: ScienceWorldRunner, args) -> tuple[list[EpisodeSpec], dict | None]:
    if args.write_episode_list:
        specs = generate_stratified_episodes(
            runner,
            args.split,
            args.k_per_task,
            seed=args.seed,
            task_names=args.tasks,
        )
        metadata = episode_list_metadata(
            specs,
            split=args.split,
            seed=args.seed,
            k_per_task=args.k_per_task,
        )
        save_episode_list(args.write_episode_list, specs, metadata)
        print(
            f"[eval] wrote {len(specs)} episodes "
            f"({metadata.task_count} tasks × up to {args.k_per_task}) "
            f"→ {args.write_episode_list}"
        )
        return specs, asdict(metadata)

    if args.episode_list:
        metadata, specs = load_episode_list(args.episode_list)
        print(
            f"[eval] loaded {len(specs)} fixed episodes "
            f"({metadata.task_count} tasks, k={metadata.k_per_task}, seed={metadata.seed})"
        )
        return specs, asdict(metadata)

    return choose_episodes(runner, args), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model")
    parser.add_argument("--main-adapter")
    parser.add_argument("--sub-adapter")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--k-per-task",
        type=int,
        default=5,
        help="For stratified lists: variations sampled per task type (default: 5).",
    )
    parser.add_argument(
        "--episode-list",
        default=None,
        help="Fixed JSON episode list (e.g. artifacts/eval/dev_stratified_k5_seed123.json).",
    )
    parser.add_argument(
        "--write-episode-list",
        default=None,
        help="Write a stratified episode list and exit (no model load).",
    )
    parser.add_argument("--step-limit", type=int, default=50)
    parser.add_argument("--max-subtasks", type=int, default=15)
    parser.add_argument(
        "--agent-interface",
        choices=("legacy", "contract-simple", "action-id", "plan-a"),
        default="legacy",
        help=(
            "legacy: subtask tags; contract-simple: minimal contract + free-form Sub; "
            "action-id: minimal contract + action-id Sub (L1); "
            "plan-a: [plan]{subgoal,focus_objects} + action-id Sub."
        ),
    )
    parser.add_argument(
        "--max-candidate-actions",
        type=int,
        default=32,
        help="For action-id: max candidate actions per step (from env valid_actions).",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=4,
        help="For action-id: recent history steps passed to Sub.",
    )
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--main-max-new-tokens", type=int, default=384)
    parser.add_argument("--sub-max-new-tokens", type=int, default=64)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="artifacts/eval/environment_eval.json")
    args = parser.parse_args()
    if args.write_episode_list:
        return args
    missing = [name for name in ("base_model", "main_adapter", "sub_adapter") if not getattr(args, name)]
    if missing:
        flag_names = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        parser.error(f"the following arguments are required: {flag_names}")
    return args


def main() -> None:
    args = parse_args()
    runner = ScienceWorldRunner(step_limit=args.step_limit)
    episode_list_info: dict | None = None
    try:
        specs, episode_list_info = resolve_episode_specs(runner, args)
        if args.write_episode_list:
            return

        policy = HierarchicalPolicy(args.base_model, args.main_adapter, args.sub_adapter, args.use_4bit)
        episodes = []
        for index, spec in enumerate(specs, 1):
            print(f"[episode {index}/{len(specs)}] {spec.task_name} variation={spec.variation_id}")
            result = run_episode(policy, runner, spec, args)
            episodes.append(result)
            print(
                f"  score={result['final_score']:.1f} "
                f"ms={result['ms_episode_score_pct']:.1f} "
                f"win={result['ms_win']} steps={result['steps']} "
                f"valid={result['action_valid_rate']:.2%}"
            )
    finally:
        runner.close()

    total_steps = sum(item["steps"] for item in episodes)
    ms = multi_square_aggregate(episodes)
    raw_mean_score = sum(item["final_score"] for item in episodes) / max(len(episodes), 1)
    report = {
        "provenance": experiment_provenance(
            {
                "base_model": args.base_model,
                "main_adapter": args.main_adapter,
                "sub_adapter": args.sub_adapter,
            }
        ),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"base_model", "main_adapter", "sub_adapter"}
        },
        "episode_list": episode_list_info,
        "metrics": {
            "scoring_protocol": "multi_square",
            "episodes": len(episodes),
            # Multi-Square primary metric: mean over episodes with ms_episode_score > 0.
            "mean_score": ms["ms_mean_score_pct"],
            "score_std": ms["ms_std_score_pct"],
            "win_rate": ms["ms_win_rate"],
            "fail_count": ms["ms_fail_count"],
            "progress_episodes": ms["ms_success_count"],
            "mean_score_all_clipped": ms["ms_mean_all_clipped_pct"],
            # Full task completion (unchanged).
            "success_rate": sum(item["success"] for item in episodes) / max(len(episodes), 1),
            # Legacy all-episode raw average (includes negatives); kept for debugging only.
            "raw_mean_score": raw_mean_score,
            "action_valid_rate": (
                sum(item["steps"] - item["invalid_actions"] for item in episodes)
                / max(total_steps, 1)
            ),
            "format_error_rate": (
                sum(item["format_errors"] for item in episodes) / max(len(episodes), 1)
            ),
            "sub_parse_error_rate": (
                sum(item.get("sub_parse_errors", 0) for item in episodes)
                / max(len(episodes), 1)
            ),
            "mean_steps": total_steps / max(len(episodes), 1),
        },
        "episodes": episodes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"[eval] wrote {output}")


if __name__ == "__main__":
    main()
