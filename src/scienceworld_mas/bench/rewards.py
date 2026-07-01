"""Official ScienceWorld reward helpers for the v2 training line."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OfficialReward:
    """Score view used for bench-faithful RL/evaluation.

    Unlike the legacy MGRPO reward helper, this object preserves negative
    ScienceWorld scores. That matters because terminal cliff failures such as
    -100 carry different information from merely failing to make progress.
    """

    raw_score: float
    normalized_score: float
    success: bool


def official_reward_from_score(score: float, *, success_threshold: float = 100.0) -> OfficialReward:
    raw = float(score)
    return OfficialReward(
        raw_score=raw,
        normalized_score=raw / success_threshold,
        success=raw >= success_threshold,
    )
