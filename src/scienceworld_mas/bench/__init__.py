"""Benchmark protocol definitions for the v2 ScienceWorld-MAS line."""

from .protocol import (
    AgentRole,
    BenchProtocol,
    DEFAULT_PROTOCOL,
    TrainingStage,
)
from .rewards import OfficialReward, official_reward_from_score

__all__ = [
    "AgentRole",
    "BenchProtocol",
    "DEFAULT_PROTOCOL",
    "OfficialReward",
    "TrainingStage",
    "official_reward_from_score",
]
