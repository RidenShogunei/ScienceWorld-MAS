"""Benchmark protocol definitions for the v2 ScienceWorld-MAS line."""

from .protocol import (
    AgentRole,
    BenchProtocol,
    DEFAULT_PROTOCOL,
    TrainingStage,
)
from .rewards import OfficialReward, official_reward_from_score
from .scoring import BenchmarkScore, EpisodeScore, TaskScore, compute_benchmark_score

__all__ = [
    "AgentRole",
    "BenchProtocol",
    "BenchmarkScore",
    "DEFAULT_PROTOCOL",
    "EpisodeScore",
    "OfficialReward",
    "TaskScore",
    "TrainingStage",
    "compute_benchmark_score",
    "official_reward_from_score",
]
