"""Audit communication-contract SFT data quality."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from contract_schema import CommunicationContract, parse_contract_text
from generate_contract_sft_data import ExpertStep, iter_expert_steps


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "called",
        "current",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if token not in stop}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def contract_to_text(contract: CommunicationContract) -> str:
    payload = contract.to_payload()
    return " ".join(
        [
            str(payload["goal"]),
            str(payload["subgoal"]),
            str(payload["rationale"]),
            " ".join(payload["target_objects"]),
            str(payload["location_hint"]),
            " ".join(payload["required_tools"]),
            str(payload["success_condition"]),
            " ".join(payload["action_guidance"]),
            str(payload["fallback_if_blocked"]),
        ]
    )


def load_main_contracts(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "val", "test"):
        for row in read_jsonl(input_dir / f"{split}.jsonl"):
            if row.get("category") == "main":
                row = dict(row)
                row["split"] = split
                rows.append(row)
    return rows


def step_key(step: ExpertStep) -> tuple[int, int]:
    return (step.source_index, step.step_index)


def row_key(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row["source_index"]), int(row["trajectory_step"]))


def audit_contracts(input_dir: Path, data_dir: Path) -> dict[str, Any]:
    steps = {step_key(step): step for step in iter_expert_steps(data_dir)}
    rows = load_main_contracts(input_dir)
    counters: Counter[str] = Counter()
    subgoal_scores: list[float] = []
    action_scores: list[float] = []
    object_scores: list[float] = []
    exact_action_coverages: list[float] = []
    exact_prefix_matches: list[bool] = []
    samples: list[dict[str, Any]] = []

    for row in rows:
        counters["main_rows"] += 1
        step = steps.get(row_key(row))
        if step is None:
            counters["missing_step"] += 1
            continue

        assistant = row["messages"][-1]["content"]
        try:
            contract = parse_contract_text(assistant)
        except Exception as exc:
            counters["parse_fail"] += 1
            samples.append(
                {
                    "source_index": row["source_index"],
                    "trajectory_step": row["trajectory_step"],
                    "issue": f"parse_fail: {exc}",
                    "assistant": assistant[:500],
                }
            )
            continue

        counters["parse_ok"] += 1
        payload = contract.to_payload()
        for field in (
            "goal",
            "subgoal",
            "rationale",
            "success_condition",
            "fallback_if_blocked",
            "action_guidance",
        ):
            if payload[field]:
                counters[f"{field}_nonempty"] += 1

        contract_text = contract_to_text(contract)
        subgoal_score = jaccard(tokens(contract.subgoal), tokens(step.subtask))
        action_score = max(
            jaccard(tokens(guidance), tokens(action))
            for guidance in contract.action_guidance
            for action in step.expert_actions
        )
        guidance_text = " || ".join(contract.action_guidance).lower()
        exact_covered = [
            action for action in step.expert_actions if action.lower() in guidance_text
        ]
        exact_action_coverage = len(exact_covered) / len(step.expert_actions)
        exact_prefix_match = contract.action_guidance[: len(step.expert_actions)] == step.expert_actions
        object_score = jaccard(
            tokens(" ".join(contract.target_objects) + " " + contract.location_hint),
            tokens(step.subtask + " " + " ".join(step.expert_actions)),
        )
        coverage_score = jaccard(tokens(contract_text), tokens(step.subtask + " " + " ".join(step.expert_actions)))
        subgoal_scores.append(subgoal_score)
        action_scores.append(action_score)
        object_scores.append(object_score)
        exact_action_coverages.append(exact_action_coverage)
        exact_prefix_matches.append(exact_prefix_match)

        weak_reasons = []
        if subgoal_score < 0.2:
            weak_reasons.append("low_subgoal_overlap")
        if action_score < 0.15:
            weak_reasons.append("low_action_guidance_overlap")
        if exact_action_coverage < 1.0:
            weak_reasons.append("missing_exact_expert_action")
        if not exact_prefix_match:
            weak_reasons.append("expert_actions_not_exact_prefix")
        if object_score == 0:
            weak_reasons.append("no_object_or_location_overlap")
        if coverage_score < 0.15:
            weak_reasons.append("low_overall_coverage")
        if weak_reasons and len(samples) < 20:
            samples.append(
                {
                    "source_index": step.source_index,
                    "trajectory_step": step.step_index,
                    "issues": weak_reasons,
                    "expert_subtask": step.subtask,
                    "expert_actions": step.expert_actions,
                    "contract": payload,
                }
            )

    total = counters["main_rows"] or 1
    parse_ok = counters["parse_ok"] or 1
    return {
        "input_dir": str(input_dir),
        "data_dir": str(data_dir),
        "main_rows": counters["main_rows"],
        "parse_ok_rate": counters["parse_ok"] / total,
        "parse_fail": counters["parse_fail"],
        "field_nonempty_rate": {
            field: counters[f"{field}_nonempty"] / parse_ok
            for field in (
                "goal",
                "subgoal",
                "rationale",
                "success_condition",
                "fallback_if_blocked",
                "action_guidance",
            )
        },
        "avg_subgoal_overlap": mean(subgoal_scores) if subgoal_scores else 0.0,
        "avg_action_guidance_overlap": mean(action_scores) if action_scores else 0.0,
        "avg_exact_expert_action_coverage": (
            mean(exact_action_coverages) if exact_action_coverages else 0.0
        ),
        "full_exact_expert_action_rate": (
            sum(1 for score in exact_action_coverages if score == 1.0) / len(exact_action_coverages)
            if exact_action_coverages
            else 0.0
        ),
        "exact_expert_action_prefix_rate": (
            sum(1 for matched in exact_prefix_matches if matched) / len(exact_prefix_matches)
            if exact_prefix_matches
            else 0.0
        ),
        "avg_object_location_overlap": mean(object_scores) if object_scores else 0.0,
        "weak_sample_count_capped": len(samples),
        "weak_samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_contracts(Path(args.input_dir), Path(args.data_dir))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
