"""Tests for Multi-Square-aligned eval metrics."""

import pytest

from eval_metrics import multi_square_aggregate, multi_square_episode_score


def test_multi_square_episode_score_clips_negative():
    assert multi_square_episode_score(-100.0) == 0.0
    assert multi_square_episode_score(0.0) == 0.0
    assert multi_square_episode_score(68.0) == 0.68


def test_multi_square_aggregate_positive_only_mean():
    episodes = [
        {"final_score": -100.0},
        {"final_score": 0.0},
        {"final_score": 68.0},
        {"final_score": 100.0},
    ]
    metrics = multi_square_aggregate(episodes)
    assert metrics["ms_fail_count"] == 2
    assert metrics["ms_success_count"] == 2
    assert metrics["ms_mean_score_pct"] == pytest.approx(84.0)
    assert metrics["ms_win_rate"] == 0.5
