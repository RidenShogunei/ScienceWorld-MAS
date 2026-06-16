"""Convert native Kimi MAS rollouts into Main/Sub SFT chat data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from collect_kimi_mas_rollouts import parse_contract_response
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
    if args.success_only and not rollout.success:
        return False
    return rollout.final_score >= args.min_final_score


def build_main_sample(rollout: SystemRollout, decision_index: int) -> dict[str, Any] | None:
    decision = rollout.main_decisions[decision_index]
    if not decision.format_valid:
        return None
    contract = parse_contract_response(decision.raw_response)
    if contract is None:
        return None
    return {
        "messages": [
            *decision.prompt_messages,
            {"role": "assistant", "content": contract.to_tagged_json()},
        ],
        "category": "main",
        "stage": "native_contract_plan",
        "source": "kimi_native_rollout",
        "rollout_id": rollout.rollout_id,
        "task_name": rollout.task_name,
        "variation_id": rollout.variation_id,
        "split": rollout.split,
        "decision_index": decision_index,
        "final_score": rollout.final_score,
        "success": rollout.success,
    }


def build_sub_samples(rollout: SystemRollout, args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    for invocation in rollout.sub_invocations:
        for step in invocation.steps:
            if not step.format_valid:
                continue
            if args.valid_actions_only and not step.action_valid:
                continue
            if not step.action:
                continue
            samples.append(
                {
                    "messages": [
                        *step.prompt_messages,
                        {
                            "role": "assistant",
                            "content": (
                                f"[action]{step.action}[/action]"
                                f"[subtask_done]{str(step.declared_subtask_done).lower()}[/subtask_done]"
                            ),
                        },
                    ],
                    "category": "sub",
                    "stage": "native_contract_act",
                    "source": "kimi_native_rollout",
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
                }
            )
    return samples


def convert_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    rollouts = [rollout for rollout in read_rollouts(Path(args.input)) if keep_rollout(rollout, args)]
    samples = []
    counts = Counter()
    for rollout in rollouts:
        for decision_index in range(len(rollout.main_decisions)):
            main_sample = build_main_sample(rollout, decision_index)
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
        "schema": "native_kimi_mas_sft_v1",
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
    return parser.parse_args()


def main() -> None:
    manifest = convert_rollouts(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
