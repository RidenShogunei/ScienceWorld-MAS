"""Evaluation metrics aligned with Multi-Square ScienceWorld scoring."""

from __future__ import annotations

import math
from typing import Any


def multi_square_episode_score(final_score: float) -> float:
    """Per-episode score used in Multi-Square ``eval_multi_sci.py`` (0–1 scale)."""
    return max(0.0, float(final_score) / 100.0)


def multi_square_aggregate(episodes: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate episodic results using Multi-Square rules.

    Matches ``alg/eval_multi_sci.py``:
    - transform: ``max(0, final_score / 100)``
    - mean/std: only episodes with transformed score > 0
    - failures: transformed score == 0 (includes clipped negatives and raw zeros)
    """
    transformed = [multi_square_episode_score(ep.get("final_score", 0.0)) for ep in episodes]
    positive = [score for score in transformed if score > 0.0]
    n = len(episodes)
    fail_count = n - len(positive)

    if positive:
        mean_pos = sum(positive) / len(positive)
        if len(positive) > 1:
            variance = sum((score - mean_pos) ** 2 for score in positive) / len(positive)
            std_pos = math.sqrt(variance)
        else:
            std_pos = 0.0
    else:
        mean_pos = 0.0
        std_pos = 0.0

    return {
        "episodes": n,
        "ms_mean_score": mean_pos,
        "ms_mean_score_pct": mean_pos * 100.0,
        "ms_std_score": std_pos,
        "ms_std_score_pct": std_pos * 100.0,
        "ms_win_rate": len(positive) / max(n, 1),
        "ms_fail_count": fail_count,
        "ms_success_count": len(positive),
        "ms_mean_all_clipped": sum(transformed) / max(n, 1),
        "ms_mean_all_clipped_pct": sum(transformed) / max(n, 1) * 100.0,
    }


def attach_multi_square_episode_fields(episode: dict[str, Any]) -> dict[str, Any]:
    """Add per-episode Multi-Square fields to an episode result dict."""
    ms = multi_square_episode_score(episode.get("final_score", 0.0))
    episode["ms_episode_score"] = ms
    episode["ms_episode_score_pct"] = ms * 100.0
    episode["ms_win"] = ms > 0.0
    return episode
