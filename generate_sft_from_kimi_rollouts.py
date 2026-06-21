"""Convert native Kimi MAS rollouts into Main/Sub SFT chat data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from collect_kimi_mas_rollouts import parse_contract_response, parse_minimal_contract_response
from generate_sft_data import write_jsonl
from rollout_schema import SystemRollout


def read_rollouts(path: Path) -> list[SystemRollout]:
    rollouts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rollouts.append(SystemRollout.from_dict(json.loads(line)))
    return rollouts


def keep_rollout(rollout: SystemRollout, args: argparse.Namespace) -> bool:
    if args.keep_local_nonnegative_steps:
        return True
    if args.success_only and not rollout.success:
        return False
    return rollout.final_score >= args.min_final_score


def invocation_has_kept_step(invocation, args: argparse.Namespace) -> bool:
    for step in invocation.steps:
        if args.valid_actions_only and not step.action_valid:
            continue
        if args.keep_local_nonnegative_steps and step.score_after < step.score_before:
            continue
        return True
    return False


def rollout_contract_schema(rollout: SystemRollout, args: argparse.Namespace) -> str:
    if args.contract_schema != "auto":
        return args.contract_schema
    if rollout.policy_version.endswith(":minimal"):
        return "minimal"
    return "verbose"


def parse_contract_for_rollout(raw_response: str, rollout: SystemRollout, args: argparse.Namespace):
    schema = rollout_contract_schema(rollout, args)
    if schema == "minimal":
        return parse_minimal_contract_response(raw_response)
    return parse_contract_response(raw_response)


def build_main_sample(rollout: SystemRollout, decision_index: int, args: argparse.Namespace) -> dict[str, Any] | None:
    decision = rollout.main_decisions[decision_index]
    if not decision.format_valid:
        return None
    if args.keep_local_nonnegative_steps and decision.invocation_id:
        invocations = {
            invocation.invocation_id: invocation for invocation in rollout.sub_invocations
        }
        invocation = invocations.get(decision.invocation_id)
        if invocation is None or not invocation_has_kept_step(invocation, args):
            return None
    contract = parse_contract_for_rollout(decision.raw_response, rollout, args)
    if contract is None:
        return None
    schema = rollout_contract_schema(rollout, args)
    return {
        "messages": [
            *decision.prompt_messages,
            {"role": "assistant", "content": contract.to_tagged_json()},
        ],
        "category": "main",
        "stage": "minimal_contract_plan" if schema == "minimal" else "contract_plan",
        "source": "kimi_native_rollout",
        "schema": "minimal_contract_v1" if schema == "minimal" else "native_kimi_mas_sft_v2",
        "rollout_id": rollout.rollout_id,
        "task_name": rollout.task_name,
        "variation_id": rollout.variation_id,
        "split": rollout.split,
        "decision_index": decision_index,
        "final_score": rollout.final_score,
        "success": rollout.success,
        "native": True,
    }


def build_sub_samples(rollout: SystemRollout, args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    seen_actions: set[str] = set()
    for invocation in rollout.sub_invocations:
        for step in invocation.steps:
            if not step.format_valid:
                continue
            if args.valid_actions_only and not step.action_valid:
                continue
            if args.keep_local_nonnegative_steps and step.score_after < step.score_before:
                continue
            if not step.action:
                continue
            normalized_action = " ".join(step.action.lower().split())
            if args.drop_repeated_actions and normalized_action in seen_actions:
                continue
            seen_actions.add(normalized_action)
            assistant_content = (
                f"[action]{step.action}[/action]"
                f"[subtask_done]{str(step.declared_subtask_done).lower()}[/subtask_done]"
                f"[handoff]{step.handoff}[/handoff]"
            )
            samples.append(
                {
                    "messages": [
                        *step.prompt_messages,
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "category": "sub",
                    "stage": (
                        "minimal_contract_act"
                        if rollout_contract_schema(rollout, args) == "minimal"
                        else "contract_act"
                    ),
                    "source": "kimi_native_rollout",
                    "schema": (
                        "minimal_contract_v1"
                        if rollout_contract_schema(rollout, args) == "minimal"
                        else "native_kimi_mas_sft_v2"
                    ),
                    "rollout_id": rollout.rollout_id,
                    "task_name": rollout.task_name,
                    "variation_id": rollout.variation_id,
                    "split": rollout.split,
                    "invocation_id": invocation.invocation_id,
                    "step_index": step.step_index,
                    "action_valid": step.action_valid,
                    "score_before": step.score_before,
                    "score_after": step.score_after,
                    "final_score": rollout.final_score,
                    "success": rollout.success,
                    "native": True,
                }
            )
    return samples


def convert_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    rollouts = [rollout for rollout in read_rollouts(Path(args.input)) if keep_rollout(rollout, args)]
    samples = []
    counts = Counter()
    for rollout in rollouts:
        for decision_index in range(len(rollout.main_decisions)):
            main_sample = build_main_sample(rollout, decision_index, args)
            if main_sample is not None:
                samples.append(main_sample)
                counts["main"] += 1
        sub_samples = build_sub_samples(rollout, args)
        samples.extend(sub_samples)
        counts["sub"] += len(sub_samples)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", samples)
    write_jsonl(output_dir / "val.jsonl", [])
    write_jsonl(output_dir / "test.jsonl", [])
    manifest = {
        "input": args.input,
        "source_rollouts": len(rollouts),
        "samples": {"train": len(samples), "val": 0, "test": 0},
        "categories": dict(counts),
        "success_only": args.success_only,
        "min_final_score": args.min_final_score,
        "valid_actions_only": args.valid_actions_only,
        "keep_local_nonnegative_steps": args.keep_local_nonnegative_steps,
        "drop_repeated_actions": args.drop_repeated_actions,
        "contract_schema": args.contract_schema,
        "schema": "native_kimi_mas_sft_v2",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/kimi_mas_sft")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--min-final-score", type=float, default=0.0)
    parser.add_argument("--valid-actions-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-local-nonnegative-steps", action="store_true")
    parser.add_argument("--drop-repeated-actions", action="store_true")
    parser.add_argument("--contract-schema", choices=("auto", "verbose", "minimal"), default="auto")
    return parser.parse_args()


def main() -> None:
    manifest = convert_rollouts(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
