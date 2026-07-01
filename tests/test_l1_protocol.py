"""Tests for L1 action-id protocol."""

from l1.protocol import (
    build_action_id_messages,
    decode_action_id,
    expert_action_id,
    parse_action_id_response,
    rank_candidate_actions,
)
from l1.config import load_config
from l1.trainer import _append_main_sample, _append_sub_sample, should_stop_after_iteration
from l1.rollout import MainCompletion, SubCompletion


def test_parse_action_id_response():
    assert parse_action_id_response("selected_action_id: 3") == 3
    assert parse_action_id_response("invalid") is None


def test_expert_action_id_and_decode():
    candidates = ["look around", "open door"]
    assert expert_action_id("open door", candidates) == 1
    assert decode_action_id(0, candidates) == "look around"


def test_build_action_id_messages_includes_contract():
    messages = build_action_id_messages(
        task="find plant",
        observation="living room",
        candidate_actions=["look around"],
        contract="[contract]{\"subgoal\":\"x\"}[/contract]",
    )
    assert "Contract:" in messages[1]["content"]


def test_rank_candidate_actions():
    actions = ["pick up egg", "look around", "open door to kitchen"]
    ranked = rank_candidate_actions(actions, context="open door kitchen", max_actions=3)
    assert len(ranked) == 3


def test_early_stop_patience():
    best, no_improve, stop, reason = should_stop_after_iteration(
        mean_expert=0.40,
        best_expert=0.35,
        iterations_without_improvement=0,
        patience=3,
        min_delta=0.01,
    )
    assert best == 0.40
    assert no_improve == 0
    assert not stop

    _, no_improve, stop, reason = should_stop_after_iteration(
        mean_expert=0.40,
        best_expert=0.40,
        iterations_without_improvement=2,
        patience=3,
        min_delta=0.01,
    )
    assert no_improve == 3
    assert stop
    assert reason == "patience"


def test_joint_config_agents_both():
    cfg = load_config("l1/config/joint.yaml")
    assert cfg.train.agents == "both"
    assert cfg.train.rollout_sub_do_sample is True


def test_append_joint_training_samples():
    main_samples: list = []
    sub_samples: list = []
    main = MainCompletion(
        raw_contract="{}",
        contract_text="{}",
        format_valid=True,
        prompt_messages=[{"role": "user", "content": "x"}],
        completion_token_ids=[1, 2],
        old_logprobs=[0.1, 0.2],
    )
    sub = SubCompletion(
        raw_response="selected_action_id: 1",
        action_id=1,
        parse_success=True,
        prompt_messages=[{"role": "user", "content": "y"}],
        completion_token_ids=[3],
        old_logprobs=[0.3],
    )
    _append_main_sample(main_samples, main, 0.5, format_valid=True, invalid_advantage=-1.0)
    _append_sub_sample(sub_samples, sub, 0.5, parse_success=False, invalid_advantage=-2.0)
    assert len(main_samples) == 1
    assert main_samples[0][2] == 0.5
    assert len(sub_samples) == 1
    assert sub_samples[0][2] == -2.0
