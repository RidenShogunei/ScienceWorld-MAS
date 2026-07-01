"""Fixed, reproducible ScienceWorld episode lists."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .scienceworld import EpisodeSpec


class VariationProvider(Protocol):
    task_names: list[str]

    def variations(self, task_name: str, split: str) -> list[int]:
        ...


@dataclass(frozen=True)
class EpisodeListMetadata:
    protocol: str
    split: str
    seed: int
    k_per_task: int
    task_count: int
    episodes: int


def generate_stratified_episodes(
    runner: VariationProvider,
    split: str,
    k_per_task: int,
    seed: int = 123,
    task_names: list[str] | None = None,
) -> list[EpisodeSpec]:
    if k_per_task <= 0:
        raise ValueError("k_per_task must be positive")

    rng = random.Random(seed)
    selected: list[EpisodeSpec] = []
    names = task_names or runner.task_names
    for task_name in sorted(names):
        variations = list(runner.variations(task_name, split))
        if not variations:
            continue
        rng.shuffle(variations)
        for variation_id in variations[: min(k_per_task, len(variations))]:
            selected.append(EpisodeSpec(task_name, int(variation_id), split))
    return selected


def episode_list_metadata(
    specs: list[EpisodeSpec],
    *,
    split: str,
    seed: int,
    k_per_task: int,
) -> EpisodeListMetadata:
    return EpisodeListMetadata(
        protocol="stratified_k_per_task",
        split=split,
        seed=seed,
        k_per_task=k_per_task,
        task_count=len({spec.task_name for spec in specs}),
        episodes=len(specs),
    )


def save_episode_list(
    path: str | Path,
    specs: list[EpisodeSpec],
    metadata: EpisodeListMetadata,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": asdict(metadata),
        "episodes": [
            {
                "task_name": spec.task_name,
                "variation_id": spec.variation_id,
                "split": spec.split,
            }
            for spec in specs
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_episode_list(path: str | Path) -> tuple[EpisodeListMetadata, list[EpisodeSpec]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = EpisodeListMetadata(**payload["metadata"])
    specs = [
        EpisodeSpec(
            task_name=item["task_name"],
            variation_id=int(item["variation_id"]),
            split=item["split"],
        )
        for item in payload["episodes"]
    ]
    if len(specs) != metadata.episodes:
        raise ValueError(
            f"episode list metadata says {metadata.episodes} episodes but file has {len(specs)}"
        )
    return metadata, specs
