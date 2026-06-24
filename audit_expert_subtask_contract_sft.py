"""Audit expert-subtask-preserving Contract SFT data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from generate_contract_sft_data import iter_expert_steps
from generate_expert_subtask_contract_sft import (
    causal_action_guidance,
    explicit_target_conflict,
)
from generate_minimal_contract_sft_data import (
    CANONICAL_HANDOFF_IF,
    parse_minimal_contract_text,
)


SUB_TARGET_PATTERN = re.compile(
    r"^\[action\](.+)\[/action\]\[subtask_done\](true|false)\[/subtask_done\]"
    r"\[handoff\](continue|complete)\[/handoff\]$",
    re.DOTALL,
)


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "val", "test"):
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_split"] = split
                rows.append(row)
    return rows


def audit(input_dir: Path, data_dir: Path) -> dict[str, Any]:
    steps = {
        (step.source_index, step.step_index): step
        for step in iter_expert_steps(data_dir)
    }
    rows = load_rows(input_dir)
    counts: Counter[str] = Counter()
    split_sources: dict[str, set[int]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    bad_samples = []

    for row in rows:
        category = row.get("category")
        counts[f"{category}_rows"] += 1
        source_index = int(row["source_index"])
        step_index = int(row["trajectory_step"])
        split_sources[row["_split"]].add(source_index)
        expert_step = steps.get((source_index, step_index))
        assistant = row["messages"][-1]["content"]

        if category == "main":
            try:
                contract = parse_minimal_contract_text(assistant)
                counts["main_parse_ok"] += 1
            except Exception as exc:
                counts["main_parse_bad"] += 1
                if len(bad_samples) < 20:
                    bad_samples.append({"issue": f"main_parse: {exc}", "row": row})
                continue
            if expert_step and contract.subgoal == expert_step.subtask.strip():
                counts["expert_subgoal_exact"] += 1
            elif len(bad_samples) < 20:
                bad_samples.append(
                    {
                        "issue": "expert_subgoal_mismatch",
                        "source_index": source_index,
                        "trajectory_step": step_index,
                        "expected": expert_step.subtask if expert_step else None,
                        "actual": contract.subgoal,
                    }
                )
            if contract.handoff_if == CANONICAL_HANDOFF_IF:
                counts["canonical_handoff"] += 1
            if expert_step and contract.action_guidance == causal_action_guidance(
                expert_step.subtask, 6
            ):
                counts["causal_guidance_exact"] += 1
            if expert_step and explicit_target_conflict(expert_step) is not None:
                counts["upstream_target_conflict"] += 1

        elif category == "sub":
            match = SUB_TARGET_PATTERN.match(assistant)
            if match:
                counts["sub_parse_ok"] += 1
                action = match.group(1)
                if expert_step:
                    action_index = int(row["action_step"])
                    if action == expert_step.expert_actions[action_index]:
                        counts["expert_action_exact"] += 1
            if "Recent execution history:" in row["messages"][-2]["content"]:
                counts["sub_has_history"] += 1
            if "Valid actions:" in row["messages"][-2]["content"]:
                counts["sub_has_valid_actions"] += 1

    main_total = counts["main_rows"] or 1
    sub_total = counts["sub_rows"] or 1
    leakage = (
        (split_sources["train"] & split_sources["val"])
        | (split_sources["train"] & split_sources["test"])
        | (split_sources["val"] & split_sources["test"])
    )
    return {
        "input_dir": str(input_dir),
        "rows": len(rows),
        "main_rows": counts["main_rows"],
        "sub_rows": counts["sub_rows"],
        "main_parse_rate": counts["main_parse_ok"] / main_total,
        "expert_subgoal_exact_rate": counts["expert_subgoal_exact"] / main_total,
        "canonical_handoff_rate": counts["canonical_handoff"] / main_total,
        "causal_guidance_exact_rate": counts["causal_guidance_exact"] / main_total,
        "upstream_target_conflict_rate": counts["upstream_target_conflict"] / main_total,
        "sub_parse_rate": counts["sub_parse_ok"] / sub_total,
        "expert_action_exact_rate": counts["expert_action_exact"] / sub_total,
        "sub_history_rate": counts["sub_has_history"] / sub_total,
        "sub_simple_prompt_rate": (
            sub_total - counts["sub_has_history"] - counts["sub_has_valid_actions"]
        )
        / sub_total,
        "sub_valid_actions_rate": counts["sub_has_valid_actions"] / sub_total,
        "source_trajectory_leakage_count": len(leakage),
        "split_source_trajectories": {
            split: len(indices) for split, indices in split_sources.items()
        },
        "bad_samples": bad_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(Path(args.input_dir), Path(args.data_dir))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
