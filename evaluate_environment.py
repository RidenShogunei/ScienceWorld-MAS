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
from generate_sft_data import MAIN_SYSTEM, SUB_SYSTEM
from provenance import experiment_provenance
from scienceworld_env import EpisodeSpec, ScienceWorldRunner
from sft_trainer import ensure_torch_set_submodule


MAIN_PATTERN = re.compile(r"\[subtask\](.*?)\[/subtask\]", re.DOTALL)
SUB_PATTERN = re.compile(
    r"\[action\](.*?)\[/action\]\s*\[subtask_done\](true|false)\[/subtask_done\]",
    re.DOTALL | re.IGNORECASE,
)


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
        state = f"Group action:{group_actions}. Current observation: {observation}"
        text = self.generate(
            "main",
            [
                {"role": "system", "content": MAIN_SYSTEM},
                {"role": "user", "content": f"Task:\n{task}\n\nPlanner state:\n{state}"},
            ],
            args.max_input_length,
            args.main_max_new_tokens,
        )
        match = MAIN_PATTERN.search(text)
        return (match.group(1).strip() if match else None), text

    def act(self, subtask: str, observation: str, args) -> tuple[str | None, bool, str]:
        text = self.generate(
            "sub",
            [
                {"role": "system", "content": SUB_SYSTEM},
                {"role": "user", "content": f"Subtask:\n{subtask}\n\nObservation:\n{observation}"},
            ],
            args.max_input_length,
            args.sub_max_new_tokens,
        )
        match = SUB_PATTERN.search(text)
        if not match:
            return None, False, text
        return match.group(1).strip(), match.group(2).lower() == "true", text


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
    observation, task, reset_info = runner.reset(spec)
    groups = []
    total_reward = 0.0
    invalid_actions = 0
    format_errors = 0
    step_count = 0
    done = False
    previous_group_actions: list[str] = []

    while not done and step_count < args.step_limit and len(groups) < args.max_subtasks:
        subtask, main_raw = policy.plan(task, observation, previous_group_actions, args)
        group = {"subtask": subtask, "main_raw": main_raw, "steps": []}
        groups.append(group)
        if subtask is None:
            format_errors += 1
            break

        current_group_actions = []
        subtask_done = False
        while not done and not subtask_done and step_count < args.step_limit:
            action, subtask_done, sub_raw = policy.act(subtask, observation, args)
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
                    "action_valid": action_valid,
                    "reward": reward,
                    "score": float(info.get("score", 0.0)),
                    "next_observation": next_observation,
                    "environment_done": done,
                }
            )
            current_group_actions.append(action)
            observation = next_observation
        previous_group_actions = current_group_actions

    final_score = 0.0
    if groups and groups[-1]["steps"]:
        final_score = groups[-1]["steps"][-1].get("score", 0.0)
    return {
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
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--main-max-new-tokens", type=int, default=64)
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
                f"  score={result['final_score']:.1f} steps={result['steps']} "
                f"valid={result['action_valid_rate']:.2%}"
            )
    finally:
        runner.close()

    total_steps = sum(item["steps"] for item in episodes)
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
            "episodes": len(episodes),
            "success_rate": sum(item["success"] for item in episodes) / max(len(episodes), 1),
            "mean_score": sum(item["final_score"] for item in episodes) / max(len(episodes), 1),
            "action_valid_rate": (
                sum(item["steps"] - item["invalid_actions"] for item in episodes)
                / max(total_steps, 1)
            ),
            "format_error_rate": (
                sum(item["format_errors"] for item in episodes) / max(len(episodes), 1)
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
