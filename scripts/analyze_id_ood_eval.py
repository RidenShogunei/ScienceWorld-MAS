#!/usr/bin/env python3
"""Build ID/OOD manifest and analyze stratified-145 eval JSONs by task coverage."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from id_ood_split import (
    build_v7_task_manifest,
    per_task_summary,
    save_manifest,
    split_episodes,
    summarize_episodes,
    training_coverage,
)
from scienceworld_env import ScienceWorldRunner

TRAIN_JSONL = ROOT / "data/expert_subtask_contract_sft_v3_simple_minimax_sample1000/train.jsonl"
EXPERT_HIGH = ROOT / "data/raw/multisquare/ScienceWorld/expert_high-data.json"
EPISODE_LIST = ROOT / "artifacts/eval/dev_stratified_k5_seed123.json"
MANIFEST_OUT = ROOT / "artifacts/eval/id_ood_manifest_v7.json"

EVAL_ROWS = [
    ("V7 Contract SFT", "artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json"),
    ("Sub-only MGRPO iter_2", "artifacts/eval/mgrpo_expert_subtask_contract_v3_v3_iter02_stratified145.json"),
    ("Sub-only MGRPO iter_10", "artifacts/eval/mgrpo_expert_subtask_contract_v3_v3_iter10_stratified145.json"),
    ("Joint MGRPO mrlx_like iter_3", "artifacts/eval/mgrpo_mrlx_like_v1_iter03_stratified145.json"),
    ("Plan A SFT smoke", "artifacts/eval/plan_a_sft_smoke_stratified145.json"),
    ("L1 Main step RL iter_10", "artifacts/eval/l1_main_step_rl_smoke_iter0010_stratified145.json"),
    ("L1 Joint step RL iter_8", "artifacts/eval/l1_joint_step_rl_iter0008_action_id_stratified145.json"),
]

TASK_CATEGORIES = {
    "find": {"find-animal", "find-living-thing", "find-non-living-thing", "find-plant"},
    "biology": {
        "grow-fruit",
        "grow-plant",
        "identify-life-stages-1",
        "identify-life-stages-2",
        "lifespan-longest-lived",
        "lifespan-longest-lived-then-shortest-lived",
        "lifespan-shortest-lived",
        "mendelian-genetics-known-plant",
        "mendelian-genetics-unknown-plant",
    },
    "incline": {
        "inclined-plane-determine-angle",
        "inclined-plane-friction-named-surfaces",
        "inclined-plane-friction-unnamed-surfaces",
    },
    "chem_phase": {
        "boil",
        "change-the-state-of-matter-of",
        "chemistry-mix",
        "chemistry-mix-paint-secondary-color",
        "chemistry-mix-paint-tertiary-color",
        "freeze",
        "melt",
    },
    "measurement": {
        "measure-melting-point-known-substance",
        "measure-melting-point-unknown-substance",
        "test-conductivity",
        "test-conductivity-of-unknown-substances",
        "use-thermometer",
    },
    "power": {"power-component", "power-component-renewable-vs-nonrenewable-energy"},
}


def category_for_task(task_name: str) -> str:
    for category, tasks in TASK_CATEGORIES.items():
        if task_name in tasks:
            return category
    return "other"


def category_breakdown(episodes: list[dict], id_tasks: set[str]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        buckets[category_for_task(ep["task_name"])].append(ep)
    rows = []
    for category in sorted(buckets):
        stats = summarize_episodes(buckets[category])
        task_names = sorted({ep["task_name"] for ep in buckets[category]})
        rows.append(
            {
                "category": category,
                "task_count": len(task_names),
                "id_task_count": sum(1 for t in task_names if t in id_tasks),
                "episodes": stats["episodes"],
                "ms_mean_score_pct": round(stats["ms_mean_score_pct"], 2),
                "ms_win_rate": round(stats["ms_win_rate"] * 100, 1),
                "success_n": stats["success_n"],
            }
        )
    return rows


def main() -> None:
    if not os.environ.get("JAVA_HOME"):
        default_java = Path("/home/jinxu/jdk-21.0.11+10-jre")
        if default_java.exists():
            os.environ["JAVA_HOME"] = str(default_java)
            os.environ["PATH"] = f"{default_java}/bin:" + os.environ.get("PATH", "")

    runner = ScienceWorldRunner()
    manifest, source_to_slug = build_v7_task_manifest(
        train_jsonl=TRAIN_JSONL,
        expert_high=EXPERT_HIGH,
        runner=runner,
    )
    save_manifest(MANIFEST_OUT, manifest, EPISODE_LIST)
    print(f"wrote {MANIFEST_OUT}")

    id_tasks = manifest.id_task_set()
    coverage = training_coverage(TRAIN_JSONL, source_to_slug)

    ms_csv = ROOT / "artifacts/eval/id_ood_ms_scores.csv"
    detail_csv = ROOT / "artifacts/eval/id_ood_eval_summary.csv"
    per_task_csv = ROOT / "artifacts/eval/id_ood_per_task_v7_sft.csv"
    category_csv = ROOT / "artifacts/eval/id_ood_category_v7_sft.csv"
    coverage_json = ROOT / "artifacts/eval/id_ood_training_coverage.json"

    coverage_json.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    ms_rows: list[dict] = []
    detail_rows: list[dict] = []
    for label, rel_path in EVAL_ROWS:
        path = ROOT / rel_path
        if not path.exists():
            continue
        episodes = json.loads(path.read_text(encoding="utf-8")).get("episodes", [])
        splits = split_episodes(episodes, id_tasks)
        id_stats = summarize_episodes(splits["id"])
        ood_stats = summarize_episodes(splits["ood"])
        all_stats = summarize_episodes(splits["all"])
        ms_rows.append(
            {
                "label": label,
                "ms_id_pct": round(id_stats["ms_mean_score_pct"], 2),
                "ms_ood_pct": round(ood_stats["ms_mean_score_pct"], 2),
                "episodes_id": id_stats["episodes"],
                "episodes_ood": ood_stats["episodes"],
            }
        )
        for split_name, split_eps in splits.items():
            stats = summarize_episodes(split_eps)
            detail_rows.append(
                {
                    "label": label,
                    "json": rel_path,
                    "split": split_name,
                    "episodes": stats["episodes"],
                    "ms_mean_score_pct": round(stats["ms_mean_score_pct"], 2),
                    "ms_win_rate_pct": round(stats["ms_win_rate"] * 100, 1),
                    "ms_win_n": stats["ms_win_n"],
                    "success_n": stats["success_n"],
                }
            )

    with ms_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "ms_id_pct", "ms_ood_pct", "episodes_id", "episodes_ood"],
        )
        writer.writeheader()
        writer.writerows(ms_rows)
    print(f"wrote {ms_csv}")

    with detail_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"wrote {detail_csv}")

    v7_path = ROOT / "artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json"
    v7_eps = json.loads(v7_path.read_text(encoding="utf-8"))["episodes"]
    task_rows = []
    for row in per_task_summary(v7_eps):
        task_rows.append(
            {
                "task_name": row["task_name"],
                "coverage": "id" if row["task_name"] in id_tasks else "ood",
                "category": category_for_task(row["task_name"]),
                "episodes": row["episodes"],
                "ms_mean_score_pct": round(row["ms_mean_score_pct"], 2),
                "ms_win_rate_pct": round(row["ms_win_rate"] * 100, 1),
                "success_n": row["success_n"],
            }
        )
    with per_task_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0].keys()))
        writer.writeheader()
        writer.writerows(task_rows)
    print(f"wrote {per_task_csv}")

    cat_rows = category_breakdown(v7_eps, id_tasks)
    with category_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cat_rows)
    print(f"wrote {category_csv}")
    print(f"wrote {coverage_json}")


if __name__ == "__main__":
    main()
