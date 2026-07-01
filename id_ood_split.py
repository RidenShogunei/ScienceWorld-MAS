"""Task-type ID/OOD grouping for ScienceWorld eval vs SFT training data."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval_metrics import multi_square_aggregate
from scienceworld_data import load_high_trajectories, strip_embedded_instruction


@dataclass(frozen=True)
class IdOodManifest:
    schema: str
    train_data: str
    expert_data: str
    definition: str
    id_tasks: tuple[str, ...]
    ood_tasks: tuple[str, ...]
    train_source_indices: tuple[int, ...]
    unmapped_train_sources: int
    unmapped_expert_trajectories: int

    def id_task_set(self) -> set[str]:
        return set(self.id_tasks)

    def ood_task_set(self) -> set[str]:
        return set(self.ood_tasks)


def _norm_task_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    return re.sub(r"\s+", " ", text).strip()


def _slug_norm_map(runner) -> dict[str, str]:
    slug_norm: dict[str, str] = {}
    for slug in runner.task_names:
        runner.env.load(slug)
        desc = strip_embedded_instruction(runner.env.get_task_description(), "Task Description:")
        slug_norm[slug] = _norm_task_text(desc)
    return slug_norm


def _map_source_index_to_slug(high_trajectories, slug_norm: dict[str, str]) -> dict[int, str]:
    norm_to_slugs: dict[str, list[str]] = defaultdict(list)
    for slug, normalized in slug_norm.items():
        norm_to_slugs[normalized].append(slug)

    mapping: dict[int, str] = {}
    unmapped = 0
    for trajectory in high_trajectories:
        task = strip_embedded_instruction(trajectory.task_description, "Task Description:")
        normalized = _norm_task_text(task)
        slugs = norm_to_slugs.get(normalized, [])
        if len(slugs) == 1:
            mapping[trajectory.source_index] = slugs[0]
            continue
        if len(slugs) > 1:
            mapping[trajectory.source_index] = slugs[0]
            continue
        hit = None
        for slug, slug_normalized in slug_norm.items():
            if normalized[:50] in slug_normalized or slug_normalized[:50] in normalized:
                hit = slug
                break
        if hit:
            mapping[trajectory.source_index] = hit
        else:
            unmapped += 1
    return mapping, unmapped


def train_source_indices(train_jsonl: Path) -> set[int]:
    indices: set[int] = set()
    for line in train_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            indices.add(json.loads(line)["source_index"])
    return indices


def build_v7_task_manifest(
    *,
    train_jsonl: Path,
    expert_high: Path,
    runner,
) -> tuple[IdOodManifest, dict[int, str]]:
    high = load_high_trajectories(expert_high)
    slug_norm = _slug_norm_map(runner)
    source_to_slug, unmapped_expert = _map_source_index_to_slug(high, slug_norm)

    train_sources = train_source_indices(train_jsonl)
    id_tasks = sorted({source_to_slug[s] for s in train_sources if s in source_to_slug})
    eval_tasks = sorted(runner.task_names)
    ood_tasks = sorted(set(eval_tasks) - set(id_tasks))

    manifest = IdOodManifest(
        schema="task_type_train_coverage_v1",
        train_data=str(train_jsonl),
        expert_data=str(expert_high),
        definition=(
            "ID = ScienceWorld task_name appears in at least one V7 SFT train source_index "
            "(Multi-Square expert trajectory mapped to task slug). "
            "OOD = dev stratified-145 task types not covered by train."
        ),
        id_tasks=tuple(id_tasks),
        ood_tasks=tuple(ood_tasks),
        train_source_indices=tuple(sorted(train_sources)),
        unmapped_train_sources=len(train_sources - set(source_to_slug)),
        unmapped_expert_trajectories=unmapped_expert,
    )
    return manifest, source_to_slug


def save_manifest(path: Path, manifest: IdOodManifest, episode_list_path: Path | None = None) -> None:
    payload: dict[str, Any] = asdict(manifest)
    if episode_list_path and episode_list_path.exists():
        episodes = json.loads(episode_list_path.read_text(encoding="utf-8"))["episodes"]
        id_set = manifest.id_task_set()
        payload["stratified145"] = {
            "episode_list": str(episode_list_path),
            "id_episodes": sum(1 for ep in episodes if ep["task_name"] in id_set),
            "ood_episodes": sum(1 for ep in episodes if ep["task_name"] not in id_set),
            "id_tasks_in_eval": sorted({ep["task_name"] for ep in episodes if ep["task_name"] in id_set}),
            "ood_tasks_in_eval": sorted({ep["task_name"] for ep in episodes if ep["task_name"] not in id_set}),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_episodes(episodes: list[dict[str, Any]], id_tasks: set[str]) -> dict[str, list[dict[str, Any]]]:
    id_eps = [ep for ep in episodes if ep.get("task_name") in id_tasks]
    ood_eps = [ep for ep in episodes if ep.get("task_name") not in id_tasks]
    return {"all": episodes, "id": id_eps, "ood": ood_eps}


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, float | int]:
    if not episodes:
        return {
            "episodes": 0,
            "raw_mean_score": 0.0,
            "ms_mean_score_pct": 0.0,
            "ms_win_rate": 0.0,
            "ms_win_n": 0,
            "success_n": 0,
            "action_valid_rate": 0.0,
            "format_error_rate": 0.0,
        }
    ms = multi_square_aggregate(episodes)
    raw_mean = sum(ep.get("final_score", 0.0) for ep in episodes) / len(episodes)
    success_n = sum(1 for ep in episodes if ep.get("final_score", 0) >= 100)
    action_valid_rates = [ep["action_valid_rate"] for ep in episodes if ep.get("action_valid_rate") is not None]
    format_errors = [ep["format_error"] for ep in episodes if ep.get("format_error") is not None]
    return {
        "episodes": len(episodes),
        "raw_mean_score": raw_mean,
        "ms_mean_score_pct": ms["ms_mean_score_pct"],
        "ms_win_rate": ms["ms_win_rate"],
        "ms_win_n": ms["ms_success_count"],
        "success_n": success_n,
        "action_valid_rate": sum(action_valid_rates) / len(action_valid_rates) if action_valid_rates else None,
        "format_error_rate": sum(format_errors) / len(format_errors) if format_errors else None,
    }


def per_task_summary(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_name"]].append(ep)
    rows = []
    for task_name in sorted(by_task):
        stats = summarize_episodes(by_task[task_name])
        rows.append({"task_name": task_name, **stats})
    return rows


def training_coverage(train_jsonl: Path, source_to_slug: dict[int, str]) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    root = train_jsonl.parent
    for split in rows_by_split:
        path = root / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows_by_split[split].append(json.loads(line))

    def split_stats(split_rows: list[dict]) -> dict[str, Any]:
        sources = {row["source_index"] for row in split_rows}
        tasks = sorted({source_to_slug[s] for s in sources if s in source_to_slug})
        main_n = sum(1 for row in split_rows if row.get("category") == "main")
        sub_n = sum(1 for row in split_rows if row.get("category") == "sub")
        return {
            "rows": len(split_rows),
            "main_rows": main_n,
            "sub_rows": sub_n,
            "source_indices": len(sources),
            "task_types": len(tasks),
            "tasks": tasks,
        }

    return {split: split_stats(rows) for split, rows in rows_by_split.items() if rows}
