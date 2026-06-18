"""Generate communication-contract SFT data from Multi-Square expert trajectories.

Kimi/Moonshot is used only to enrich Main-to-Sub communication contracts.
Low-level action labels remain the official Multi-Square expert actions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from contract_schema import CommunicationContract, build_mock_contract, parse_contract_text
from generate_sft_data import write_jsonl
from scienceworld_data import (
    assign_split,
    load_high_trajectories,
    load_low_trajectories,
    parse_action_done,
    strip_embedded_instruction,
    task_family,
)


CONTRACT_MAIN_SYSTEM = (
    "You are the main planning agent in ScienceWorld. Communicate with the executor using "
    "a grounded JSON contract, not a one-line subtask. Output exactly one "
    "[contract]{...}[/contract] block."
)

CONTRACT_SUB_SYSTEM = (
    "You are the ScienceWorld executor. Use the structured contract and current observation "
    "to produce one executable action and whether the contract subgoal is complete. Output "
    "exactly: [action]...[/action][subtask_done]true|false[/subtask_done]"
    "[handoff]continue|complete|blocked|need_replan[/handoff]"
)


DISTILL_SYSTEM = (
    "You convert ScienceWorld expert trajectories into grounded Main-to-Sub communication "
    "contracts. Return only strict JSON with keys: goal, subgoal, rationale, target_objects, "
    "location_hint, required_tools, success_condition, action_guidance, fallback_if_blocked. "
    "The fields target_objects, required_tools, and action_guidance must be JSON arrays of strings. "
    "The action_guidance array must begin with every expert action string exactly as provided, "
    "in the same order. You may append short extra guidance after those exact strings. "
    "Do not invent or rewrite expert action labels."
)


@dataclass(frozen=True)
class ExpertStep:
    source_index: int
    step_index: int
    task: str
    observation: str
    subtask: str
    expert_actions: list[str]
    low_observations: list[str]
    low_dones: list[bool]
    task_family: str
    split_key: str


def iter_expert_steps(data_dir: Path, limit: int | None = None) -> list[ExpertStep]:
    high = load_high_trajectories(data_dir / "expert_high-data.json")
    low = load_low_trajectories(data_dir / "expert_low-data.json")
    flat_high = []
    for trajectory in high:
        task = strip_embedded_instruction(trajectory.task_description, "Task Description:")
        family = task_family(trajectory.task_description)
        for step_index, (observation, subtask) in enumerate(
            zip(trajectory.observations, trajectory.subtasks)
        ):
            flat_high.append((trajectory.source_index, step_index, task, observation, subtask, family))

    if len(flat_high) != len(low):
        raise ValueError(f"high/low step count mismatch: {len(flat_high)} high vs {len(low)} low")

    steps = []
    for item, low_trajectory in zip(flat_high, low):
        source_index, step_index, task, observation, subtask, family = item
        low_subtask = strip_embedded_instruction(low_trajectory.subtask_prompt, "Subtask:")
        if low_subtask.strip().lower() != subtask.strip().lower():
            raise ValueError(
                f"subtask mismatch at source={source_index} step={step_index}: "
                f"{subtask!r} != {low_subtask!r}"
            )
        expert_actions = [parse_action_done(value)[0] for value in low_trajectory.actions]
        steps.append(
            ExpertStep(
                source_index=source_index,
                step_index=step_index,
                task=task,
                observation=observation,
                subtask=subtask,
                expert_actions=expert_actions,
                low_observations=low_trajectory.observations,
                low_dones=low_trajectory.dones,
                task_family=family,
                split_key=family,
            )
        )
        if limit is not None and len(steps) >= limit:
            break
    return steps


def distill_contract_with_kimi(step: ExpertStep, args: argparse.Namespace) -> CommunicationContract:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai>=1.0 to use --provider kimi") from exc

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.api_base)
    user_prompt = {
        "task": step.task,
        "planner_observation": step.observation,
        "expert_subtask": step.subtask,
        "expert_actions": step.expert_actions,
        "executor_observations": step.low_observations[:3],
    }
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": DISTILL_SYSTEM},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    content = response.choices[0].message.content or ""
    return parse_contract_text(content)


def extract_first_json_object(text: str) -> str:
    """Return the first balanced JSON object embedded in CLI output."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    raise ValueError("no JSON object found in Kimi Code CLI output")


def _assistant_content_from_stream_json(stdout: str) -> str:
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


def distill_contract_with_kimicode_cli(
    step: ExpertStep,
    args: argparse.Namespace,
) -> CommunicationContract:
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

    user_prompt = {
        "task": step.task,
        "planner_observation": step.observation,
        "expert_subtask": step.subtask,
        "expert_actions": step.expert_actions,
        "executor_observations": step.low_observations[:3],
    }
    prompt = (
        "You are a JSON API for ScienceWorld contract distillation.\n"
        f"{DISTILL_SYSTEM}\n"
        "Return one strict JSON object only. Do not use markdown. Do not explain.\n\n"
        f"Input:\n{json.dumps(user_prompt, ensure_ascii=False)}"
    )
    env = os.environ.copy()
    env.update(
        {
            "KIMI_MODEL_NAME": args.kimicode_model,
            "KIMI_MODEL_API_KEY": api_key,
            "KIMI_MODEL_PROVIDER_TYPE": args.kimicode_provider_type,
            "KIMI_MODEL_BASE_URL": args.kimicode_base_url,
            "KIMI_MODEL_TEMPERATURE": str(args.temperature),
            "KIMI_DISABLE_TELEMETRY": "1",
        }
    )
    completed = subprocess.run(
        [cli_path, "-p", prompt, "--output-format", "stream-json"],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=args.kimicode_timeout,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"Kimi Code CLI failed with exit code {completed.returncode}: {stderr}")

    content = _assistant_content_from_stream_json(completed.stdout)
    return parse_contract_text(extract_first_json_object(content))


def distill_contract_with_kimicode_http(
    step: ExpertStep,
    args: argparse.Namespace,
) -> CommunicationContract:
    api_key = os.environ.get(args.kimicode_api_key_env)
    if not api_key and args.kimicode_api_key_file:
        api_key = Path(args.kimicode_api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {args.kimicode_api_key_env} or pass --kimicode-api-key-file."
        )

    user_prompt = {
        "task": step.task,
        "planner_observation": step.observation,
        "expert_subtask": step.subtask,
        "expert_actions": step.expert_actions,
        "executor_observations": step.low_observations[:3],
    }
    prompt = (
        "You are a JSON API for ScienceWorld contract distillation.\n"
        f"{DISTILL_SYSTEM}\n"
        "Return one strict JSON object only. Do not use markdown. Do not explain.\n\n"
        f"Input:\n{json.dumps(user_prompt, ensure_ascii=False)}"
    )
    payload = {
        "model": args.kimicode_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    endpoint = args.kimicode_base_url.rstrip("/") + "/v1/messages"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
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
    return parse_contract_text(extract_first_json_object(content))


def align_contract_to_expert_actions(
    contract: CommunicationContract,
    expert_actions: list[str],
) -> CommunicationContract:
    """Keep Kimi's rich fields but force action guidance onto official labels."""
    extras = []
    expert_lower = [action.lower() for action in expert_actions]
    for item in contract.action_guidance:
        normalized = item.lower()
        if normalized in expert_lower:
            continue
        if any(action in normalized or normalized in action for action in expert_lower):
            continue
        extras.append(item)
    return CommunicationContract(
        goal=contract.goal,
        subgoal=contract.subgoal,
        rationale=contract.rationale,
        target_objects=contract.target_objects,
        location_hint=contract.location_hint,
        required_tools=contract.required_tools,
        success_condition=contract.success_condition,
        action_guidance=[*expert_actions, *extras],
        fallback_if_blocked=contract.fallback_if_blocked,
    )


def build_main_sample(step: ExpertStep, contract: CommunicationContract) -> dict:
    return {
        "messages": [
            {"role": "system", "content": CONTRACT_MAIN_SYSTEM},
            {
                "role": "user",
                "content": f"Task:\n{step.task}\n\nPlanner state:\n{step.observation}",
            },
            {"role": "assistant", "content": contract.to_tagged_json()},
        ],
        "category": "main",
        "stage": "contract_plan",
        "source_index": step.source_index,
        "trajectory_step": step.step_index,
        "task_family": step.task_family,
        "distilled": True,
    }


def build_sub_samples(step: ExpertStep, contract: CommunicationContract) -> list[dict]:
    samples = []
    for idx, encoded_action in enumerate(step.expert_actions):
        done = step.low_dones[idx]
        handoff = "complete" if done else "continue"
        observation = step.low_observations[idx]
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": CONTRACT_SUB_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Contract:\n{contract.to_tagged_json()}\n\nObservation:\n{observation}",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"[action]{encoded_action}[/action]"
                            f"[subtask_done]{str(done).lower()}[/subtask_done]"
                            f"[handoff]{handoff}[/handoff]"
                        ),
                    },
                ],
                "category": "sub",
                "stage": "contract_act",
                "source_index": step.source_index,
                "trajectory_step": step.step_index,
                "action_step": idx,
                "task_family": step.task_family,
                "distilled": True,
            }
        )
    return samples


def contract_for_step(step: ExpertStep, args: argparse.Namespace) -> CommunicationContract:
    cache_path = Path(args.cache_dir) / f"{step.source_index}_{step.step_index}.json"
    if args.cache_dir and cache_path.exists():
        contract = parse_contract_text(cache_path.read_text(encoding="utf-8"))
        return align_contract_to_expert_actions(contract, step.expert_actions)

    if args.provider == "mock":
        contract = build_mock_contract(
            task=step.task,
            subtask=step.subtask,
            expert_actions=step.expert_actions,
            observation=step.observation,
        )
        if args.cache_dir:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(contract.to_payload(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return contract
    for attempt in range(args.retries + 1):
        try:
            if args.provider == "kimi":
                contract = distill_contract_with_kimi(step, args)
            elif args.provider == "kimicode-cli":
                contract = distill_contract_with_kimicode_cli(step, args)
            elif args.provider == "kimicode-http":
                contract = distill_contract_with_kimicode_http(step, args)
            else:
                raise ValueError(f"unsupported provider: {args.provider}")
            contract = align_contract_to_expert_actions(contract, step.expert_actions)
            if args.cache_dir:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(contract.to_payload(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            return contract
        except Exception as exc:
            failure_path = Path(args.failure_dir) / f"{step.source_index}_{step.step_index}.txt"
            if args.failure_dir:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(repr(exc), encoding="utf-8")
            if exc.__class__.__name__ == "AuthenticationError":
                raise RuntimeError(
                    "Kimi authentication failed. Check that the key belongs to the "
                    "Moonshot/Kimi OpenAI-compatible Chat Completions API."
                ) from exc
            if "401" in str(exc) or "403" in str(exc):
                raise RuntimeError(
                    "Kimi request failed with an authentication/authorization error. "
                    "For Kimi Code keys, use --provider kimicode-cli and set "
                    "KIMI_CODE_API_KEY."
                ) from exc
            if attempt >= args.retries:
                raise
            time.sleep(args.retry_sleep)
    raise RuntimeError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--output-dir", default="data/contract_sft")
    parser.add_argument("--provider", choices=("mock", "kimi", "kimicode-cli", "kimicode-http"), default="mock")
    parser.add_argument("--model", default="kimi-k2.6")
    parser.add_argument("--api-base", default="https://api.moonshot.ai/v1")
    parser.add_argument("--api-key-env", default="MOONSHOT_API_KEY")
    parser.add_argument("--kimicode-model", default="kimi-for-coding")
    parser.add_argument("--kimicode-base-url", default="https://api.kimi.com/coding")
    parser.add_argument("--kimicode-provider-type", default="kimi")
    parser.add_argument("--kimicode-api-key-env", default="KIMI_CODE_API_KEY")
    parser.add_argument("--kimicode-api-key-file", default="")
    parser.add_argument("--kimicode-cli-path", default="")
    parser.add_argument("--kimicode-timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--cache-dir", default="artifacts/contract_distill_cache")
    parser.add_argument("--failure-dir", default="artifacts/contract_distill_failures")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = iter_expert_steps(Path(args.data_dir), args.limit)
    by_split = {split: [] for split in ("train", "val", "test")}
    counts = Counter()
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
            print(
                "[contract-sft] skipped "
                f"source={step.source_index} step={step.step_index}: {exc}"
            )
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
            print(f"[contract-sft] distilled {index}/{len(steps)} expert steps")

    output_dir = Path(args.output_dir)
    for split, samples in by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", samples)
    manifest = {
        "provider": args.provider,
        "model": args.kimicode_model if args.provider in {"kimicode-cli", "kimicode-http"} else args.model,
        "api_base": (
            args.kimicode_base_url
            if args.provider in {"kimicode-cli", "kimicode-http"}
            else args.api_base
            if args.provider != "mock"
            else None
        ),
        "expert_steps": len(steps),
        "samples": {split: len(samples) for split, samples in by_split.items()},
        "categories": dict(counts),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "schema": "contract_v2",
        "skipped": len(skipped),
    }
    if skipped:
        manifest["skipped_items"] = skipped
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
