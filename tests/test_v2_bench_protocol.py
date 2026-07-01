from scienceworld_mas.bench import DEFAULT_PROTOCOL, TrainingStage, official_reward_from_score


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
