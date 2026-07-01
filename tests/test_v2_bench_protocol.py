from scienceworld_mas.bench import (
    DEFAULT_PROTOCOL,
    EpisodeScore,
    TrainingStage,
    compute_benchmark_score,
    official_reward_from_score,
)


def test_default_protocol_is_bench_faithful():
    DEFAULT_PROTOCOL.validate()
    assert DEFAULT_PROTOCOL.metric_primary == "official_mean_score"
    assert DEFAULT_PROTOCOL.use_official_score_for_rl
    assert DEFAULT_PROTOCOL.preserve_negative_scores
    assert DEFAULT_PROTOCOL.system1_train_stages == (TrainingStage.SYSTEM1_SFT,)
    assert TrainingStage.JOINT_RL_ABLATION in DEFAULT_PROTOCOL.ablation_stages


def test_official_reward_preserves_negative_scores():
    reward = official_reward_from_score(-100.0)
    assert reward.raw_score == -100.0
    assert reward.normalized_score == -1.0
    assert not reward.success


def test_official_reward_marks_success_at_100():
    reward = official_reward_from_score(100.0)
    assert reward.raw_score == 100.0
    assert reward.normalized_score == 1.0
    assert reward.success


def test_benchmark_score_matches_official_episode_mean():
    score = compute_benchmark_score(
        [
            EpisodeScore("boil", 0, 100.0, steps=12, action_valid_count=10, action_count=12),
            EpisodeScore("boil", 1, -100.0, steps=4, action_valid_count=1, action_count=4),
            EpisodeScore("freeze", 0, 50.0, steps=8, action_valid_count=8, action_count=8),
        ]
    )
    assert score.episodes == 3
    assert score.official_mean_score == 50.0 / 3.0
    assert score.success_rate == 1.0 / 3.0
    assert score.negative_score_rate == 1.0 / 3.0
    assert score.action_valid_rate == 19.0 / 24.0
    assert score.mean_steps == 8.0


def test_benchmark_score_reports_task_means():
    score = compute_benchmark_score(
        [
            EpisodeScore("b-task", 0, 100.0, steps=1),
            EpisodeScore("a-task", 0, -100.0, steps=1),
            EpisodeScore("a-task", 1, 50.0, steps=1),
        ]
    )
    assert [item.task_name for item in score.score_by_task] == ["a-task", "b-task"]
    assert score.score_by_task[0].official_mean_score == -25.0
    assert score.score_by_task[1].official_mean_score == 100.0
