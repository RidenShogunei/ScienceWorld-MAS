import pytest
import torch

from mgrpo_batch import build_mgrpo_batch
from mgrpo_objective import add_reference_kl, clipped_policy_loss
from rollout_schema import ActionStep, MainDecision, SubInvocation, SystemRollout
from scienceworld_rewards import main_reward, sub_invocation_reward
from trajectory_alignment import align_sub_invocations, group_relative_advantages
from mgrpo_trainer import sample_iter_specs
from rollout_schema import group_key
from scienceworld_env import EpisodeSpec


def make_step(
    index: int,
    action: str = "look around",
    valid: bool = True,
    score_before: float = 0.0,
    score_after: float = 0.0,
    declared_done: bool = False,
    environment_done: bool = False,
) -> ActionStep:
    return ActionStep(
        step_index=index,
        observation=f"observation {index}",
        raw_response=(
            f"[action]{action}[/action]"
            f"[subtask_done]{str(declared_done).lower()}[/subtask_done]"
        ),
        action=action,
        format_valid=True,
        action_valid=valid,
        declared_subtask_done=declared_done,
        environment_reward=score_after - score_before,
        score_before=score_before,
        score_after=score_after,
        next_observation=f"observation {index + 1}",
        environment_done=environment_done,
    )


def make_rollout(
    rollout_id: str,
    final_score: float,
    invocation_count: int = 1,
    repeated: bool = False,
    valid: bool = True,
) -> SystemRollout:
    decisions = []
    invocations = []
    score = 0.0
    for index in range(invocation_count):
        invocation_id = f"{rollout_id}:sub:{index}"
        subtask = "repeat task" if repeated else f"subtask {index}"
        next_score = final_score if index == invocation_count - 1 else score
        step = make_step(
            0,
            action="valid action" if valid else "invented action",
            valid=valid,
            score_before=score,
            score_after=next_score,
            declared_done=next_score <= score,
            environment_done=next_score >= 100,
        )
        decisions.append(
            MainDecision(
                decision_index=index,
                observation=f"state {index}",
                previous_group_actions=[],
                raw_response=f"[subtask]{subtask}[/subtask]",
                subtask=subtask,
                format_valid=True,
                score_before=score,
                invocation_id=invocation_id,
            )
        )
        invocations.append(
            SubInvocation(
                invocation_id=invocation_id,
                parent_main_index=index,
                subtask=subtask,
                steps=[step],
            )
        )
        score = next_score

    rollout = SystemRollout(
        rollout_id=rollout_id,
        group_key="dev:boil:0",
        task_name="boil",
        variation_id=0,
        split="dev",
        task_description="Boil water",
        policy_version="test-policy",
        main_decisions=decisions,
        sub_invocations=invocations,
        final_score=final_score,
        environment_done=final_score >= 100,
    )
    rollout.validate()
    return rollout


def test_rollout_round_trip():
    rollout = make_rollout("r0", 100.0, invocation_count=2)
    restored = SystemRollout.from_dict(rollout.to_dict())
    assert restored == rollout
    assert restored.action_steps[-1].score_delta == 100.0


def test_group_key_includes_variation_id():
    assert group_key("boil", 7, "dev") == "dev:boil:7"
    assert group_key("boil", 0, "dev") != group_key("boil", 1, "dev")


def test_sample_iter_specs_repeats_each_group():
    pool = [
        EpisodeSpec("task-a", 1, "dev"),
        EpisodeSpec("task-b", 2, "dev"),
        EpisodeSpec("task-c", 3, "dev"),
    ]
    specs = sample_iter_specs(pool, groups=2, group_size=4, seed=1)
    assert len(specs) == 8
    from collections import Counter

    counts = Counter(group_key(s.task_name, s.variation_id, s.split) for s in specs)
    assert len(counts) == 2
    assert all(value == 4 for value in counts.values())


def test_group_advantages_are_nonzero_with_repeated_query_rollouts():
    rollouts = [
        make_rollout("a", 0.0),
        make_rollout("b", 50.0),
        make_rollout("c", 100.0),
    ]
    advantages = group_relative_advantages(rollouts, lambda item: item.final_score)
    assert advantages["a"] < 0
    assert advantages["c"] > 0
    assert any(abs(value) > 1e-6 for value in advantages.values())


def test_group_advantages_are_query_relative():
    rollouts = [
        make_rollout("low", 0.0),
        make_rollout("mid", 50.0),
        make_rollout("high", 100.0),
    ]
    advantages = group_relative_advantages(rollouts, lambda item: item.final_score)
    assert advantages["low"] < 0
    assert advantages["mid"] == pytest.approx(0.0, abs=1e-6)
    assert advantages["high"] > 0
    assert sum(advantages.values()) == pytest.approx(0.0, abs=1e-6)


def test_equal_rewards_have_zero_advantage():
    rollouts = [make_rollout("a", 0.0), make_rollout("b", 0.0)]
    assert group_relative_advantages(rollouts, lambda item: item.final_score) == {
        "a": 0.0,
        "b": 0.0,
    }


def test_alignment_masks_zero_sub_rollout():
    rollout = make_rollout("direct", 0.0, invocation_count=0)
    aligned = align_sub_invocations([rollout], target_invocations=3)
    assert len(aligned[0].slots) == 3
    assert all(slot.invocation is None for slot in aligned[0].slots)
    assert all(slot.loss_mask == 0.0 for slot in aligned[0].slots)


def test_alignment_duplicates_indices_without_mutating_trajectory():
    rollout = make_rollout("short", 0.0, invocation_count=2)
    original_ids = [id(item) for item in rollout.sub_invocations]
    aligned = align_sub_invocations([rollout], target_invocations=5, seed=7)[0]
    assert len(aligned.slots) == 5
    assert all(slot.loss_mask == 1.0 for slot in aligned.slots)
    assert sum(slot.duplicated for slot in aligned.slots) == 3
    assert {slot.source_invocation_index for slot in aligned.slots} == {0, 1}
    assert [id(item) for item in rollout.sub_invocations] == original_ids
    assert all(
        slot.invocation is rollout.sub_invocations[slot.source_invocation_index]
        for slot in aligned.slots
    )


def test_alignment_downsamples_long_rollout_deterministically():
    rollout = make_rollout("long", 0.0, invocation_count=6)
    first = align_sub_invocations([rollout], target_invocations=3, seed=9)[0]
    second = align_sub_invocations([rollout], target_invocations=3, seed=9)[0]
    assert [slot.source_invocation_index for slot in first.slots] == [
        slot.source_invocation_index for slot in second.slots
    ]
    assert len({slot.source_invocation_index for slot in first.slots}) == 3
    assert not any(slot.duplicated for slot in first.slots)


def test_successful_progressing_rollout_outscores_repeated_failure():
    success = make_rollout("success", 100.0, invocation_count=2, valid=True)
    failure = make_rollout("failure", 0.0, invocation_count=2, repeated=True, valid=False)
    assert main_reward(success).total > main_reward(failure).total
    assert (
        sub_invocation_reward(success, success.sub_invocations[-1]).total
        > sub_invocation_reward(failure, failure.sub_invocations[-1]).total
    )


def test_mgrpo_batch_normalizes_main_and_aligned_sub_separately():
    rollouts = [
        make_rollout("low", 0.0, invocation_count=1, valid=False),
        make_rollout("high", 100.0, invocation_count=2, valid=True),
    ]
    batch = build_mgrpo_batch(rollouts, target_invocations=3, seed=5)
    assert len(batch.main_records) == 2
    assert len(batch.sub_records) == 6
    assert sum(item.advantage for item in batch.main_records) == pytest.approx(0.0, abs=1e-6)
    assert sum(item.advantage for item in batch.sub_records) == pytest.approx(0.0, abs=1e-6)
    assert max(item.advantage for item in batch.sub_records) > 0
    assert min(item.advantage for item in batch.sub_records) < 0


def test_mgrpo_batch_keeps_empty_sub_slots_masked():
    direct = make_rollout("direct", 0.0, invocation_count=0)
    delegated = make_rollout("delegated", 100.0, invocation_count=1)
    batch = build_mgrpo_batch([direct, delegated], target_invocations=2)
    direct_slots = [item for item in batch.sub_records if item.rollout_id == "direct"]
    assert len(direct_slots) == 2
    assert all(item.loss_mask == 0.0 for item in direct_slots)
    assert all(item.reward == 0.0 and item.advantage == 0.0 for item in direct_slots)


def test_strict_format_gate_zeroes_reward():
    rollout = make_rollout("bad-format", 100.0)
    rollout.main_decisions[0].format_valid = False
    rollout.sub_invocations[0].steps[0].format_valid = False
    assert main_reward(rollout).total == 0.0
    assert sub_invocation_reward(rollout, rollout.sub_invocations[0]).total == 0.0


def test_first_decision_format_penalty():
    from scienceworld_rewards import RewardWeights

    rollout = make_rollout("bad-first", 100.0)
    rollout.main_decisions[0].format_valid = False
    weights = RewardWeights(
        strict_format_gate=False,
        first_decision_format_penalty=0.3,
    )
    breakdown = main_reward(rollout, weights)
    assert breakdown.components["first_decision_format_penalty"] == -0.3


def test_collect_main_training_samples_penalizes_invalid_format():
    from mgrpo_trainer import collect_main_training_samples

    rollout = make_rollout("mixed", 50.0)
    rollout.main_decisions[0].format_valid = False
    rollout.main_decisions[0].completion_token_ids = [1, 2, 3]
    rollout.main_decisions[0].old_logprobs = [-0.1, -0.2, -0.3]
    samples = collect_main_training_samples(
        [rollout],
        {rollout.rollout_id: 0.5},
        invalid_format_advantage=-1.0,
    )
    assert len(samples) == 1
    assert samples[0][2] == -1.0


def test_clipped_policy_loss_uses_token_and_slot_masks():
    current = torch.log(torch.tensor([[1.5, 1.0], [3.0, 3.0]]))
    old = torch.zeros_like(current)
    advantages = torch.tensor([1.0, 10.0])
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    loss, metrics = clipped_policy_loss(current, old, advantages, mask)
    assert loss.item() == pytest.approx(-1.1, abs=1e-6)
    assert metrics["active_tokens"].item() == 2
    assert metrics["clip_fraction"].item() == pytest.approx(0.5)


def test_reference_kl_is_added_to_policy_loss():
    policy_loss = torch.tensor(2.0)
    current = torch.tensor([[0.2, 0.4]])
    reference = torch.tensor([[0.1, 0.1]])
    total, sampled_kl = add_reference_kl(
        policy_loss,
        current,
        reference,
        torch.ones_like(current),
        beta=0.5,
    )
    assert sampled_kl.item() == pytest.approx(0.0228275, abs=1e-6)
    assert total.item() == pytest.approx(2.0114138, abs=1e-6)
