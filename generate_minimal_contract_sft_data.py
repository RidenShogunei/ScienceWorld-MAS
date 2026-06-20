"""Generate minimal contract-native SFT data from expert trajectories.

This pipeline keeps executable action guidance under program control. Kimi may
distill semantic contract fields, but the low-level action labels always come
from the official expert trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_contract_sft_data import ExpertStep, extract_first_json_object, iter_expert_steps
from generate_sft_data import write_jsonl
from scienceworld_data import assign_split


MINIMAL_CONTRACT_KEYS = (
    "subgoal",
    "success_condition",
    "target_objects",
    "action_guidance",
    "handoff_if",
)

MINIMAL_MAIN_SYSTEM = (
    "You are the main planning agent in ScienceWorld. Give the executor a compact "
    "JSON contract with the exact interface it should follow. Output exactly one "
    "[contract]{...}[/contract] block."
)

MINIMAL_SUB_SYSTEM = (
    "You are the ScienceWorld executor. Use the contract and current observation to "
    "produce one executable environment action. Output exactly: "
    "[action]...[/action][subtask_done]true|false[/subtask_done]"
    "[handoff]continue|complete|blocked|need_replan[/handoff]"
)

MINIMAL_DISTILL_SYSTEM = (
    "You convert ScienceWorld expert trajectory context into a compact executor "
    "contract. Return only strict JSON with keys: subgoal, success_condition, "
    "target_objects. target_objects must be an array of strings. "
    "Do not include action_guidance, rationale, fallback_if_blocked, location_hint, "
    "required_tools, goal, handoff_if, markdown, or explanations."
)

CANONICAL_HANDOFF_IF = "complete when success_condition is met; need_replan if blocked"


@dataclass(frozen=True)
class MinimalContract:
    subgoal: str
    success_condition: str
    target_objects: list[str]
    action_guidance: list[str]
    handoff_if: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "subgoal": self.subgoal.strip(),
            "success_condition": self.success_condition.strip(),
            "target_objects": [item.strip() for item in self.target_objects if item.strip()],
            "action_guidance": [item.strip() for item in self.action_guidance if item.strip()],
            "handoff_if": self.handoff_if.strip(),
        }

    def to_tagged_json(self) -> str:
        return "[contract]" + json.dumps(self.to_payload(), ensure_ascii=False, separators=(",", ":")) + "[/contract]"


def unique_action_guidance(expert_actions: list[str], limit: int) -> list[str]:
    seen = set()
    guidance = []
    for action in expert_actions:
        normalized = action.strip()
        if normalized.isdigit():
            continue
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        guidance.append(normalized)
        if len(guidance) >= limit:
            break
    return guidance


def parse_minimal_contract_text(text: str) -> MinimalContract:
    stripped = text.strip()
    if stripped.startswith("[contract]"):
        stripped = stripped[len("[contract]") :]
    if stripped.endswith("[/contract]"):
        stripped = stripped[: -len("[/contract]")]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("minimal contract must be a JSON object")
    unexpected = sorted(set(payload) - set(MINIMAL_CONTRACT_KEYS))
    if unexpected:
        raise ValueError(f"unexpected minimal contract keys: {unexpected}")
    missing = [key for key in MINIMAL_CONTRACT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"missing minimal contract keys: {missing}")
    target_objects = payload["target_objects"]
    action_guidance = payload["action_guidance"]
    if not isinstance(target_objects, list) or not all(isinstance(item, str) for item in target_objects):
        raise ValueError("target_objects must be a list of strings")
    if not isinstance(action_guidance, list) or not all(isinstance(item, str) for item in action_guidance):
        raise ValueError("action_guidance must be a list of strings")
    return MinimalContract(
        subgoal=str(payload["subgoal"]),
        success_condition=str(payload["success_condition"]),
        target_objects=target_objects,
        action_guidance=action_guidance,
        handoff_if=str(payload["handoff_if"]),
    )


def minimal_mock_contract(step: ExpertStep, guidance_limit: int) -> MinimalContract:
    return MinimalContract(
        subgoal=step.subtask,
        success_condition=(
            "The executor has completed the subgoal and should hand control back to Main."
        ),
        target_objects=extract_target_objects(step.subtask, step.expert_actions),
        action_guidance=unique_action_guidance(step.expert_actions, guidance_limit),
        handoff_if=CANONICAL_HANDOFF_IF,
    )


def extract_target_objects(subtask: str, expert_actions: list[str]) -> list[str]:
    candidates = []
    for text in [subtask, *expert_actions]:
        for marker in ("the ", "a ", "an "):
            if marker in text.lower():
                tail = text.lower().split(marker, 1)[1]
                phrase = tail.split(" in ")[0].split(" on ")[0].split(" to ")[0].strip(" .")
                if phrase and len(phrase.split()) <= 5:
                    candidates.append(phrase)
    cleaned = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            cleaned.append(item)
    return cleaned[:5]


def _kimicode_api_key(args: argparse.Namespace) -> str:
    api_key = os.environ.get(args.kimicode_api_key_env, "")
    if not api_key and args.kimicode_api_key_file:
        api_key = Path(args.kimicode_api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {args.kimicode_api_key_env} or pass --kimicode-api-key-file."
        )
    return api_key


def distill_semantics_with_kimicode_http(step: ExpertStep, args: argparse.Namespace) -> dict[str, Any]:
    user_prompt = {
        "task": step.task,
        "planner_observation": step.observation,
        "expert_subtask": step.subtask,
        "expert_actions": step.expert_actions,
        "executor_observations": step.low_observations[:3],
    }
    prompt = (
        "You are a JSON API for ScienceWorld contract distillation.\n"
        f"{MINIMAL_DISTILL_SYSTEM}\n"
        "Return one strict JSON object only. Do not use markdown. Do not explain.\n\n"
        f"Input:\n{json.dumps(user_prompt, ensure_ascii=False)}"
    )
    payload = {
        "model": args.kimicode_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    request = urllib.request.Request(
        args.kimicode_base_url.rstrip("/") + "/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": _kimicode_api_key(args),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.kimicode_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kimi Code HTTP failed with HTTP {exc.code}: {body}") from exc
    content = "\n".join(
        str(chunk.get("text", ""))
        for chunk in data.get("content", [])
        if isinstance(chunk, dict) and chunk.get("type") == "text"
    )
    parsed = json.loads(extract_first_json_object(content))
    if not isinstance(parsed, dict):
        raise ValueError("Kimi semantic contract must be a JSON object")
    return parsed


def contract_for_step(step: ExpertStep, args: argparse.Namespace) -> MinimalContract:
    cache_path = Path(args.cache_dir) / f"{step.source_index}_{step.step_index}.json"
    if args.cache_dir and cache_path.exists():
        return parse_minimal_contract_text(cache_path.read_text(encoding="utf-8"))

    if args.provider == "mock":
        contract = minimal_mock_contract(step, args.guidance_limit)
    elif args.provider == "kimicode-http":
        semantics = distill_semantics_with_retry(step, args)
        unexpected = sorted(set(semantics) - {"subgoal", "success_condition", "target_objects"})
        if unexpected:
            raise ValueError(f"Kimi returned forbidden keys: {unexpected}")
        target_objects = semantics.get("target_objects", [])
        if not isinstance(target_objects, list):
            target_objects = []
        contract = MinimalContract(
            subgoal=str(semantics.get("subgoal") or step.subtask),
            success_condition=str(
                semantics.get("success_condition")
                or "The executor has completed the requested subgoal."
            ),
            target_objects=[str(item) for item in target_objects],
            action_guidance=unique_action_guidance(step.expert_actions, args.guidance_limit),
            handoff_if=CANONICAL_HANDOFF_IF,
        )
    else:
        raise ValueError(f"unsupported provider: {args.provider}")

    contract = parse_minimal_contract_text(contract.to_tagged_json())
    if args.cache_dir:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(contract.to_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return contract


def distill_semantics_with_retry(step: ExpertStep, args: argparse.Namespace) -> dict[str, Any]:
    for attempt in range(args.retries + 1):
        try:
            return distill_semantics_with_kimicode_http(step, args)
        except Exception as exc:
            failure_path = Path(args.failure_dir) / f"{step.source_index}_{step.step_index}.txt"
            if args.failure_dir:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(repr(exc), encoding="utf-8")
            if attempt >= args.retries:
                raise
            time.sleep(args.retry_sleep)
    raise RuntimeError("unreachable")


def build_main_sample(step: ExpertStep, contract: MinimalContract) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": MINIMAL_MAIN_SYSTEM},
            {
                "role": "user",
                "content": f"Task:\n{step.task}\n\nPlanner state:\n{step.observation}",
            },
            {"role": "assistant", "content": contract.to_tagged_json()},
        ],
        "category": "main",
        "stage": "minimal_contract_plan",
        "source_index": step.source_index,
        "trajectory_step": step.step_index,
        "task_family": step.task_family,
        "schema": "minimal_contract_v1",
        "distilled": True,
    }


def build_sub_samples(step: ExpertStep, contract: MinimalContract) -> list[dict[str, Any]]:
    samples = []
    for idx, action in enumerate(step.expert_actions):
        done = step.low_dones[idx]
        handoff = "complete" if done else "continue"
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": MINIMAL_SUB_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Contract:\n{contract.to_tagged_json()}\n\n"
                            f"Observation:\n{step.low_observations[idx]}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"[action]{action}[/action]"
                            f"[subtask_done]{str(done).lower()}[/subtask_done]"
                            f"[handoff]{handoff}[/handoff]"
                        ),
                    },
                ],
                "category": "sub",
                "stage": "minimal_contract_act",
                "source_index": step.source_index,
                "trajectory_step": step.step_index,
                "action_step": idx,
                "task_family": step.task_family,
                "schema": "minimal_contract_v1",
                "distilled": True,
            }
        )
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--output-dir", default="data/minimal_contract_sft")
    parser.add_argument("--provider", choices=("mock", "kimicode-http"), default="mock")
    parser.add_argument("--kimicode-model", default="kimi-for-coding")
    parser.add_argument("--kimicode-base-url", default="https://api.kimi.com/coding")
    parser.add_argument("--kimicode-api-key-env", default="KIMI_CODE_API_KEY")
    parser.add_argument("--kimicode-api-key-file", default="")
    parser.add_argument("--kimicode-timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--cache-dir", default="artifacts/minimal_contract_distill_cache")
    parser.add_argument("--failure-dir", default="artifacts/minimal_contract_distill_failures")
    parser.add_argument("--guidance-limit", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Deterministically sample this many expert steps across the full dataset.",
    )
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = iter_expert_steps(Path(args.data_dir), None if args.sample_size else args.limit)
    if args.sample_size is not None:
        rng = random.Random(args.seed)
        if args.sample_size > len(steps):
            raise ValueError(f"--sample-size {args.sample_size} exceeds available steps {len(steps)}")
        steps = rng.sample(steps, args.sample_size)
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "val", "test")}
    counts: Counter[str] = Counter()
    skipped = []

    for index, step in enumerate(steps, 1):
        try:
            contract = contract_for_step(step, args)
        except Exception as exc:
            if not args.skip_failures:
                raise
            skipped.append(
                {
                    "source_index": step.source_index,
                    "trajectory_step": step.step_index,
                    "error": repr(exc),
                }
            )
            print(f"[minimal-contract-sft] skipped source={step.source_index} step={step.step_index}: {exc}")
            continue

        split = assign_split(
            step.split_key,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        samples = [build_main_sample(step, contract)]
        if not args.main_only:
            samples.extend(build_sub_samples(step, contract))
        by_split[split].extend(samples)
        counts.update(sample["category"] for sample in samples)
        if index % 25 == 0:
            print(f"[minimal-contract-sft] distilled {index}/{len(steps)} expert steps")

    output_dir = Path(args.output_dir)
    all_samples = []
    for split, samples in by_split.items():
        all_samples.extend(samples)
        write_jsonl(output_dir / f"{split}.jsonl", samples)
    write_jsonl(output_dir / "all.jsonl", all_samples)

    manifest = {
        "schema": "minimal_contract_v1",
        "provider": args.provider,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "guidance_limit": args.guidance_limit,
        "expert_steps_requested": len(steps),
        "expert_steps_generated": len(steps) - len(skipped),
        "skipped": skipped,
        "samples": {split: len(samples) for split, samples in by_split.items()},
        "all_samples": len(all_samples),
        "categories": dict(counts),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
