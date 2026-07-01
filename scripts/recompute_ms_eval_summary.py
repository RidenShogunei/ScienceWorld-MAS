#!/usr/bin/env python3
"""Offline recompute Multi-Square metrics from stratified-145 eval JSON files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_metrics import multi_square_aggregate

ROWS = [
    {
        "group": "contract",
        "label": "V7 Contract SFT",
        "experiment": "sft_expert_subtask_contract_v3",
        "protocol": "contract-simple",
        "checkpoint": "artifacts/checkpoints/sft_expert_subtask_contract_v3/",
        "json": "artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json",
    },
    {
        "group": "contract",
        "label": "Sub-only MGRPO v3 iter_2",
        "experiment": "mgrpo_expert_subtask_contract_v3_sub_only_v3/iter_0002",
        "protocol": "contract-simple",
        "checkpoint": "Main=SFT best, Sub=iter_0002/sub",
        "json": "artifacts/eval/mgrpo_expert_subtask_contract_v3_v3_iter02_stratified145.json",
    },
    {
        "group": "contract",
        "label": "Sub-only MGRPO v3 iter_10",
        "experiment": "mgrpo_expert_subtask_contract_v3_sub_only_v3/iter_0010",
        "protocol": "contract-simple",
        "checkpoint": "Main=SFT best, Sub=iter_0010/sub",
        "json": "artifacts/eval/mgrpo_expert_subtask_contract_v3_v3_iter10_stratified145.json",
    },
    {
        "group": "contract",
        "label": "Joint MGRPO mrlx_like iter_3",
        "experiment": "mgrpo_expert_subtask_contract_v3_mrlx_like_v1/iter_0003",
        "protocol": "contract-simple",
        "checkpoint": "Main+Sub=iter_0003",
        "json": "artifacts/eval/mgrpo_mrlx_like_v1_iter03_stratified145.json",
    },
]


def load_row(row: dict) -> dict:
    path = ROOT / row["json"]
    data = json.loads(path.read_text())
    episodes = data.get("episodes", [])
    metrics = data.get("metrics", {})
    ms = multi_square_aggregate(episodes)
    success_n = sum(1 for ep in episodes if ep.get("final_score", 0) >= 100)
    return {
        **row,
        "episodes": len(episodes),
        "raw_mean_score": metrics.get("mean_score"),
        "success_rate": metrics.get("success_rate"),
        "success_n": success_n,
        "action_valid_rate": metrics.get("action_valid_rate"),
        "format_error_rate": metrics.get("format_error_rate"),
        "mean_steps": metrics.get("mean_steps"),
        "ms_mean_score_pct": ms["ms_mean_score_pct"],
        "ms_win_rate": ms["ms_win_rate"],
        "ms_win_n": ms["ms_success_count"],
        "ms_fail_count": ms["ms_fail_count"],
        "ms_mean_all_clipped_pct": ms["ms_mean_all_clipped_pct"],
    }


def main() -> None:
    results = [load_row(row) for row in ROWS]
    out_csv = ROOT / "artifacts/eval/ms_eval_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "label",
        "experiment",
        "protocol",
        "json",
        "episodes",
        "raw_mean_score",
        "ms_mean_score_pct",
        "ms_mean_all_clipped_pct",
        "ms_win_rate",
        "ms_win_n",
        "ms_fail_count",
        "success_rate",
        "success_n",
        "action_valid_rate",
        "format_error_rate",
        "mean_steps",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {out_csv}")
    for r in results:
        print(
            f"{r['label']}: raw={r['raw_mean_score']:.2f} "
            f"MS={r['ms_mean_score_pct']:.1f}% win={r['ms_win_rate']*100:.1f}% "
            f"success={r['success_n']}/145"
        )


if __name__ == "__main__":
    main()
