"""ScienceWorld benchmark scoring.

This module is intentionally small and strict: it defines how v2 turns episode
records into benchmark metrics. Environment runners and training code should
call this rather than each inventing their own score aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EpisodeScore:
    task_name: str
    variation_id: int
    score: float
    steps: int
    action_valid_count: int = 0
    action_count: int = 0
    format_error_count: int = 0

    @property
    def success(self) -> bool:
        return self.score >= 100.0

    @property
    def negative(self) -> bool:
        return self.score < 0.0


@dataclass(frozen=True)
class TaskScore:
    task_name: str
    episodes: int
    official_mean_score: float
    success_rate: float


@dataclass(frozen=True)
class BenchmarkScore:
    episodes: int
    official_mean_score: float
    success_rate: float
    negative_score_rate: float
    action_valid_rate: float
    format_error_rate: float
    mean_steps: float
    score_by_task: tuple[TaskScore, ...]

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "official_mean_score": self.official_mean_score,
            "success_rate": self.success_rate,
            "negative_score_rate": self.negative_score_rate,
            "action_valid_rate": self.action_valid_rate,
            "format_error_rate": self.format_error_rate,
            "mean_steps": self.mean_steps,
            "score_by_task": [
                {
                    "task_name": item.task_name,
                    "episodes": item.episodes,
                    "official_mean_score": item.official_mean_score,
                    "success_rate": item.success_rate,
                }
                for item in self.score_by_task
            ],
        }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def compute_benchmark_score(episodes: Iterable[EpisodeScore]) -> BenchmarkScore:
    items = list(episodes)
    if not items:
        return BenchmarkScore(
            episodes=0,
            official_mean_score=0.0,
            success_rate=0.0,
            negative_score_rate=0.0,
            action_valid_rate=0.0,
            format_error_rate=0.0,
            mean_steps=0.0,
            score_by_task=(),
        )

    total_actions = sum(item.action_count for item in items)
    total_format_errors = sum(item.format_error_count for item in items)
    by_task: dict[str, list[EpisodeScore]] = {}
    for item in items:
        by_task.setdefault(item.task_name, []).append(item)

    task_scores = tuple(
        TaskScore(
            task_name=task_name,
            episodes=len(task_items),
            official_mean_score=_mean(item.score for item in task_items),
            success_rate=_mean(1.0 if item.success else 0.0 for item in task_items),
        )
        for task_name, task_items in sorted(by_task.items())
    )

    return BenchmarkScore(
        episodes=len(items),
        official_mean_score=_mean(item.score for item in items),
        success_rate=_mean(1.0 if item.success else 0.0 for item in items),
        negative_score_rate=_mean(1.0 if item.negative else 0.0 for item in items),
        action_valid_rate=(
            sum(item.action_valid_count for item in items) / total_actions
            if total_actions
            else 0.0
        ),
        format_error_rate=total_format_errors / len(items),
        mean_steps=_mean(float(item.steps) for item in items),
        score_by_task=task_scores,
    )
