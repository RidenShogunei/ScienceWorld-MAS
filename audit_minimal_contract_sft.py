"""Audit minimal contract-native SFT data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from generate_contract_sft_data import ExpertStep, iter_expert_steps
from generate_minimal_contract_sft_data import (
    CANONICAL_HANDOFF_IF,
    MINIMAL_CONTRACT_KEYS,
    parse_minimal_contract_text,
)


FORBIDDEN_KEYS = {
    "goal",
    "rationale",
    "location_hint",
    "required_tools",
    "fallback_if_blocked",
}
SUB_TARGET_PATTERN = re.compile(
    r"^\[action\].+\[/action\]\[subtask_done\](true|false)\[/subtask_done\]"
    r"\[handoff\](continue|complete|blocked|need_replan)\[/handoff\]$"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def step_key(step: ExpertStep) -> tuple[int, int]:
    return (step.source_index, step.step_index)


def row_key(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row["source_index"]), int(row["trajectory_step"]))


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
    return sorted(values)[index]


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "val", "test"):
        for row in read_jsonl(input_dir / f"{split}.jsonl"):
            item = dict(row)
            item["split"] = split
            rows.append(item)
    return rows


def audit(input_dir: Path, data_dir: Path, guidance_limit: int) -> dict[str, Any]:
    steps = {step_key(step): step for step in iter_expert_steps(data_dir)}
    rows = load_rows(input_dir)
    counters: Counter[str] = Counter()
    guidance_lengths: list[int] = []
    assistant_chars: list[int] = []
    bad_samples: list[dict[str, Any]] = []

    for row in rows:
        category = row.get("category")
        counters[f"{category}_rows"] += 1
        assistant = row["messages"][-1]["content"]
        assistant_chars.append(len(assistant))

        if category == "main":
            step = steps.get(row_key(row))
            counters["main_total"] += 1
            try:
                contract = parse_minimal_contract_text(assistant)
            except Exception as exc:
                counters["main_parse_fail"] += 1
                if len(bad_samples) < 20:
                    bad_samples.append(
                        {
                            "issue": f"parse_fail: {exc}",
                            "source_index": row.get("source_index"),
                            "trajectory_step": row.get("trajectory_step"),
                            "assistant": assistant[:500],
                        }
                    )
                continue

            counters["main_parse_ok"] += 1
            payload = contract.to_payload()
            keys = set(payload)
            if keys == set(MINIMAL_CONTRACT_KEYS):
                counters["main_exact_keyset"] += 1
            if keys & FORBIDDEN_KEYS:
                counters["main_forbidden_keys"] += 1
            for field in MINIMAL_CONTRACT_KEYS:
                if payload[field]:
                    counters[f"{field}_nonempty"] += 1
            guidance_lengths.append(len(contract.action_guidance))
            if len(contract.action_guidance) <= guidance_limit:
                counters["guidance_under_limit"] += 1
            if contract.handoff_if == CANONICAL_HANDOFF_IF:
                counters["canonical_handoff_if"] += 1
            if step is not None:
                expected = []
                seen = set()
                for action in step.expert_actions:
                    if action.strip().isdigit():
                        continue
                    key = action.lower()
                    if key not in seen:
                        seen.add(key)
                        expected.append(action)
                    if len(expected) >= guidance_limit:
                        break
                if contract.action_guidance == expected:
                    counters["guidance_exact_programmatic"] += 1
                else:
                    counters["guidance_mismatch"] += 1
                    if len(bad_samples) < 20:
                        bad_samples.append(
                            {
                                "issue": "guidance_mismatch",
                                "source_index": row.get("source_index"),
                                "trajectory_step": row.get("trajectory_step"),
                                "expected": expected,
                                "actual": contract.action_guidance,
                            }
                        )

        elif category == "sub":
            counters["sub_total"] += 1
            if SUB_TARGET_PATTERN.match(assistant):
                counters["sub_target_ok"] += 1
            elif len(bad_samples) < 20:
                counters["sub_target_bad"] += 1
                bad_samples.append(
                    {
                        "issue": "sub_target_bad",
                        "source_index": row.get("source_index"),
                        "trajectory_step": row.get("trajectory_step"),
                        "action_step": row.get("action_step"),
                        "assistant": assistant,
                    }
                )

    main_total = counters["main_total"] or 1
    sub_total = counters["sub_total"] or 1
    parse_ok = counters["main_parse_ok"] or 1
    return {
        "input_dir": str(input_dir),
        "data_dir": str(data_dir),
        "rows": len(rows),
        "main_rows": counters["main_total"],
        "sub_rows": counters["sub_total"],
        "main_parse_ok_rate": counters["main_parse_ok"] / main_total,
        "main_exact_keyset_rate": counters["main_exact_keyset"] / parse_ok,
        "main_forbidden_key_rate": counters["main_forbidden_keys"] / parse_ok,
        "sub_target_ok_rate": counters["sub_target_ok"] / sub_total,
        "guidance_exact_programmatic_rate": counters["guidance_exact_programmatic"] / parse_ok,
        "guidance_under_limit_rate": counters["guidance_under_limit"] / parse_ok,
        "canonical_handoff_if_rate": counters["canonical_handoff_if"] / parse_ok,
        "field_nonempty_rate": {
            field: counters[f"{field}_nonempty"] / parse_ok for field in MINIMAL_CONTRACT_KEYS
        },
        "guidance_length": {
            "avg": mean(guidance_lengths) if guidance_lengths else 0.0,
            "p95": percentile(guidance_lengths, 0.95),
            "max": max(guidance_lengths) if guidance_lengths else 0,
        },
        "assistant_chars": {
            "avg": mean(assistant_chars) if assistant_chars else 0.0,
            "p95": percentile(assistant_chars, 0.95),
            "max": max(assistant_chars) if assistant_chars else 0,
        },
        "counters": dict(counters),
        "bad_samples": bad_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--guidance-limit", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(Path(args.input_dir), Path(args.data_dir), args.guidance_limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
