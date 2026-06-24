"""Generate causal Contract SFT from Multi-Square expert subtask boundaries.

The expert high-level subtask is the immutable planning label. An optional LLM
may enrich success_condition and target_objects, but it cannot replace or
rewrite the subgoal. Low-level actions, completion labels, and handoff labels
remain deterministic expert supervision.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from generate_contract_sft_data import ExpertStep, extract_first_json_object, iter_expert_steps
from generate_minimal_contract_sft_data import (
    CANONICAL_HANDOFF_IF,
    MINIMAL_MAIN_SYSTEM,
    MINIMAL_SUB_SYSTEM,
    MinimalContract,
    _minimax_api_key,
    clean_contract_text,
    extract_target_objects,
    strip_thinking,
)
from generate_sft_data import write_jsonl
from scienceworld_data import assign_split


ENRICH_SYSTEM = (
    "You enrich an immutable ScienceWorld expert subtask with execution metadata. "
    "Return strict JSON with exactly two keys: success_condition and target_objects. "
    "success_condition must describe an observable condition proving that the provided "
    "expert_subtask is complete. target_objects must be an array of strings grounded in "
    "the expert_subtask or supplied observations. Do not create a new subgoal, plan a "
    "different task, include actions, or use information not present in the input."
)


def mock_enrichment(step: ExpertStep) -> dict[str, Any]:
    return {
        "success_condition": (
            f"The executor has completed the expert subtask: {step.subtask.strip()}"
        ),
        "target_objects": extract_target_objects(step.subtask, []),
    }


def minimax_enrichment(step: ExpertStep, args: argparse.Namespace) -> dict[str, Any]:
    input_payload = {
        "task": step.task,
        "planner_observation": step.observation,
        "expert_subtask": step.subtask,
    }
    prompt = (
        f"{ENRICH_SYSTEM}\n"
        "Return one JSON object only. Do not use markdown or explanations.\n\n"
        f"Input:\n{json.dumps(input_payload, ensure_ascii=False)}"
    )
    payload: dict[str, Any] = {
        "model": args.minimax_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.minimax_model == "MiniMax-M3":
        payload["thinking"] = {"type": "disabled"}
    request = urllib.request.Request(
        args.minimax_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_minimax_api_key(args)}",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiniMax HTTP failed with HTTP {exc.code}: {body}") from exc
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("MiniMax response did not include choices")
    content = strip_thinking(str(choices[0].get("message", {}).get("content", "")))
    parsed = json.loads(extract_first_json_object(content))
    if not isinstance(parsed, dict):
        raise ValueError("enrichment must be a JSON object")
    if set(parsed) != {"success_condition", "target_objects"}:
        raise ValueError(f"unexpected enrichment keys: {sorted(parsed)}")
    if not isinstance(parsed["target_objects"], list):
        raise ValueError("target_objects must be an array")
    return parsed


def enrichment_for_step(step: ExpertStep, args: argparse.Namespace) -> dict[str, Any]:
    cache_path = Path(args.cache_dir) / f"{step.source_index}_{step.step_index}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if set(cached) != {"success_condition", "target_objects"}:
            raise ValueError(f"invalid cache schema: {cache_path}")
        return cached

    if args.provider == "mock":
        enrichment = mock_enrichment(step)
    else:
        for attempt in range(args.retries + 1):
            try:
                enrichment = minimax_enrichment(step, args)
                break
            except Exception as exc:
                if attempt >= args.retries:
                    raise
                delay = args.retry_sleep * (2**attempt)
                if "HTTP 429" in str(exc):
                    delay = max(delay, args.rate_limit_sleep)
                time.sleep(delay)

    normalized = {
        "success_condition": clean_contract_text(
            str(enrichment.get("success_condition") or "").strip()
        ),
        "target_objects": [
            clean_contract_text(str(item).strip())
            for item in enrichment.get("target_objects", [])
            if str(item).strip()
        ][: args.target_object_limit],
    }
    if not normalized["success_condition"]:
        normalized["success_condition"] = mock_enrichment(step)["success_condition"]
    if not normalized["target_objects"]:
        normalized["target_objects"] = extract_target_objects(
            step.subtask, []
        )[: args.target_object_limit]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized


def contract_for_step(step: ExpertStep, enrichment: dict[str, Any], args: argparse.Namespace) -> MinimalContract:
    contract = MinimalContract(
        # This field is deliberately not sourced from the enrichment model.
        subgoal=clean_contract_text(step.subtask.strip()),
        success_condition=str(enrichment["success_condition"]),
        target_objects=[str(item) for item in enrichment["target_objects"]],
        action_guidance=causal_action_guidance(step.subtask, args.guidance_limit),
        handoff_if=CANONICAL_HANDOFF_IF,
    )
    return contract


def causal_action_guidance(subtask: str, limit: int) -> list[str]:
    """Build guidance from the expert subtask without reading future actions."""
    text = " ".join(subtask.strip().split())
    lowered = text.lower()
    guidance = []
    rules = (
        (("navigate", "navigation", "go to", "travel"), f"Use available navigation actions to complete: {text}"),
        (("find", "locate", "search"), f"Inspect relevant locations and containers to complete: {text}"),
        (("prepare", "get ", "collect", "pick up", "take "), f"Collect the objects required to complete: {text}"),
        (("move ", "place ", "put ", "fill "), f"Manipulate the named objects to complete: {text}"),
        (("heat", "boil", "melt", "freeze", "cool"), f"Operate the relevant equipment and monitor the state change for: {text}"),
        (("measure", "determine", "test "), f"Use the required measurement or experiment procedure for: {text}"),
        (("connect", "circuit", "power "), f"Connect and operate the named circuit components for: {text}"),
        (("focus",), f"Use a focus action on the target named by: {text}"),
        (("wait", "monitor", "grow"), f"Observe and advance the environment until this condition is met: {text}"),
        (("read", "recipe", "instructions"), f"Inspect or read the relevant information for: {text}"),
    )
    for keywords, instruction in rules:
        if any(keyword in lowered for keyword in keywords):
            guidance.append(instruction)
        if len(guidance) >= limit:
            break
    return guidance or [f"Choose executable actions that directly complete: {text}"]


BOX_TARGET_PATTERN = re.compile(
    r"\b(?:to|in)\s+(?:the\s+)?"
    r"(red|green|blue|orange|yellow|purple|violet|black|white|brown|grey|gray)"
    r"\s+box\b",
    re.IGNORECASE,
)


def explicit_target_conflict(step: ExpertStep) -> str | None:
    """Detect unambiguous upstream box-target contradictions."""
    subtask_targets = {value.lower() for value in BOX_TARGET_PATTERN.findall(step.subtask)}
    action_targets = {
        value.lower()
        for value in BOX_TARGET_PATTERN.findall(" ".join(step.expert_actions))
    }
    if subtask_targets and action_targets and subtask_targets.isdisjoint(action_targets):
        return (
            f"subtask targets {sorted(subtask_targets)} but expert actions target "
            f"{sorted(action_targets)}"
        )
    return None


def main_sample(step: ExpertStep, contract: MinimalContract) -> dict[str, Any]:
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
        "stage": "expert_subtask_contract_plan",
        "source": "multisquare_expert_subtask",
        "schema": "expert_subtask_contract_v1",
        "source_index": step.source_index,
        "trajectory_step": step.step_index,
        "task_family": step.task_family,
        "expert_subgoal": step.subtask.strip(),
        "causal_subgoal": True,
    }


def recent_history(step: ExpertStep, action_index: int, history_limit: int) -> list[dict[str, Any]]:
    start = max(0, action_index - history_limit)
    rows = []
    for index in range(start, action_index):
        rows.append(
            {
                "action": step.expert_actions[index],
                "subtask_done": step.low_dones[index],
            }
        )
    return rows


def sub_samples(step: ExpertStep, contract: MinimalContract, args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    for action_index, action in enumerate(step.expert_actions):
        done = step.low_dones[action_index]
        handoff = "complete" if done else "continue"
        user_content = (
            f"Contract:\n{contract.to_tagged_json()}\n\n"
            f"Observation:\n{step.low_observations[action_index]}"
        )
        if args.include_history:
            history = recent_history(step, action_index, args.history_limit)
            user_content += (
                "\n\nRecent execution history:\n"
                + json.dumps(history, ensure_ascii=False, indent=2)
            )
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": MINIMAL_SUB_SYSTEM},
                    {"role": "user", "content": user_content},
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
                "stage": "expert_subtask_contract_act",
                "source": "multisquare_expert_action",
                "schema": "expert_subtask_contract_v1",
                "source_index": step.source_index,
                "trajectory_step": step.step_index,
                "action_step": action_index,
                "task_family": step.task_family,
                "expert_subgoal": step.subtask.strip(),
                "valid_actions_available": False,
                "history_available": args.include_history,
            }
        )
    return samples


def split_for_step(step: ExpertStep, args: argparse.Namespace) -> str:
    # All high-level decisions and their low-level actions from one expert
    # episode stay in the same split.
    return assign_split(
        f"high_source:{step.source_index}",
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


def generate_step(
    step: ExpertStep, args: argparse.Namespace
) -> tuple[ExpertStep, MinimalContract, list[dict[str, Any]]]:
    enrichment = enrichment_for_step(step, args)
    contract = contract_for_step(step, enrichment, args)
    rows = [main_sample(step, contract)]
    if not args.main_only:
        rows.extend(sub_samples(step, contract, args))
    return step, contract, rows


def choose_steps(args: argparse.Namespace) -> list[ExpertStep]:
    steps = iter_expert_steps(Path(args.data_dir))
    if args.drop_explicit_target_conflicts:
        steps = [step for step in steps if explicit_target_conflict(step) is None]
    if args.sample_size is None:
        return steps
    if args.sample_size > len(steps):
        raise ValueError(
            f"--sample-size {args.sample_size} exceeds available steps {len(steps)}"
        )
    rng = random.Random(args.seed)
    return rng.sample(steps, args.sample_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument(
        "--output-dir", default="data/expert_subtask_contract_sft_v1_sample1000"
    )
    parser.add_argument("--provider", choices=("mock", "minimax"), default="mock")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--guidance-limit", type=int, default=6)
    parser.add_argument("--target-object-limit", type=int, default=5)
    parser.add_argument("--history-limit", type=int, default=6)
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--cache-dir", default="artifacts/expert_subtask_contract_cache"
    )
    parser.add_argument("--minimax-model", default="MiniMax-M3")
    parser.add_argument("--minimax-base-url", default="https://api.minimaxi.com/v1")
    parser.add_argument("--minimax-api-key-env", default="MINIMAX_API_KEY")
    parser.add_argument("--minimax-api-key-file", default="")
    parser.add_argument("--kimicode-api-key-env", default="KIMI_CODE_API_KEY")
    parser.add_argument("--kimicode-api-key-file", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--rate-limit-sleep", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    steps = choose_steps(args)
    all_steps = iter_expert_steps(Path(args.data_dir))
    upstream_conflicts = [
        {
            "source_index": step.source_index,
            "trajectory_step": step.step_index,
            "reason": explicit_target_conflict(step),
        }
        for step in all_steps
        if explicit_target_conflict(step) is not None
    ]
    results: list[tuple[ExpertStep, MinimalContract, list[dict[str, Any]]] | None] = [
        None
    ] * len(steps)

    if args.workers == 1:
        for index, step in enumerate(steps):
            results[index] = generate_step(step, args)
            if (index + 1) % 25 == 0:
                print(f"[expert-subtask-contract] generated {index + 1}/{len(steps)}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_step, step, args): index
                for index, step in enumerate(steps)
            }
            completed = 0
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                completed += 1
                if completed % 25 == 0:
                    print(f"[expert-subtask-contract] generated {completed}/{len(steps)}")

    by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_sources: dict[str, set[int]] = {key: set() for key in by_split}
    category_counts: Counter[str] = Counter()
    exact_subgoal = 0
    task_families: set[str] = set()
    for result in results:
        if result is None:
            raise RuntimeError("missing generation result")
        step, contract, rows = result
        split = split_for_step(step, args)
        by_split[split].extend(rows)
        split_sources[split].add(step.source_index)
        category_counts.update(row["category"] for row in rows)
        exact_subgoal += int(contract.subgoal == clean_contract_text(step.subtask.strip()))
        task_families.add(step.task_family)

    output_dir = Path(args.output_dir)
    all_rows = []
    for split, rows in by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        all_rows.extend(rows)
    write_jsonl(output_dir / "all.jsonl", all_rows)

    leakage = (
        (split_sources["train"] & split_sources["val"])
        | (split_sources["train"] & split_sources["test"])
        | (split_sources["val"] & split_sources["test"])
    )
    manifest = {
        "schema": "expert_subtask_contract_v1",
        "provider": args.provider,
        "data_design": {
            "main_subgoal_source": "immutable_multisquare_expert_subtask",
            "success_condition_source": args.provider,
            "target_objects_source": args.provider,
            "main_enrichment_inputs": [
                "task",
                "planner_observation",
                "immutable_expert_subtask",
            ],
            "action_guidance_source": "programmatic_expert_subtask_only",
            "sub_action_source": "multisquare_expert_actions",
            "sub_recent_history": args.include_history,
            "sub_valid_actions": False,
            "split_group": "high_level_source_trajectory",
        },
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "expert_steps": len(steps),
        "upstream_explicit_target_conflicts_dropped": (
            len(upstream_conflicts) if args.drop_explicit_target_conflicts else 0
        ),
        "upstream_explicit_target_conflict_examples": upstream_conflicts[:20],
        "task_families": len(task_families),
        "samples": {split: len(rows) for split, rows in by_split.items()},
        "categories": dict(category_counts),
        "source_trajectories": {
            split: len(indices) for split, indices in split_sources.items()
        },
        "source_trajectory_leakage_count": len(leakage),
        "expert_subgoal_exact_rate": exact_subgoal / max(len(steps), 1),
        "all_samples": len(all_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
