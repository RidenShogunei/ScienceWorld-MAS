"""Build synchronized Main/Sub M-GRPO training records."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rollout_schema import SubInvocation, SystemRollout
from scienceworld_rewards import RewardBreakdown, RewardWeights, main_reward, sub_invocation_reward
from trajectory_alignment import AlignedSubSlot, align_sub_invocations, group_relative_advantages


@dataclass(frozen=True)
class MainTrainingRecord:
    rollout_id: str
    group_key: str
    reward: float
    advantage: float
    breakdown: RewardBreakdown


@dataclass(frozen=True)
class SubTrainingRecord:
    rollout_id: str
    group_key: str
    slot_index: int
    invocation: SubInvocation | None
    source_invocation_index: int | None
    reward: float
    advantage: float
    loss_mask: float
    duplicated: bool
    breakdown: RewardBreakdown | None


@dataclass(frozen=True)
class MGRPOBatch:
    main_records: tuple[MainTrainingRecord, ...]
    sub_records: tuple[SubTrainingRecord, ...]
    target_invocations: int


def _normalize_sub_rewards(
    slots_with_rewards: list[tuple[AlignedSubSlot, float, RewardBreakdown | None]],
    epsilon: float,
) -> dict[tuple[str, int], float]:
    grouped: dict[str, list[tuple[AlignedSubSlot, float]]] = {}
    for slot, reward, _ in slots_with_rewards:
        if slot.loss_mask:
            grouped.setdefault(slot.group_key, []).append((slot, reward))

    advantages = {}
    for key, members in grouped.items():
        rewards = [reward for _, reward in members]
        if not all(math.isfinite(value) for value in rewards):
            raise ValueError(f"Sub group {key} contains a non-finite reward")
        mean = sum(rewards) / len(rewards)
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        std = math.sqrt(variance)
        for slot, reward in members:
            advantages[(slot.rollout_id, slot.slot_index)] = (
                (reward - mean) / (std + epsilon) if std > epsilon else 0.0
            )
    return advantages


def build_mgrpo_batch(
    rollouts: list[SystemRollout],
    target_invocations: int,
    seed: int = 123,
    reward_weights: RewardWeights = RewardWeights(),
    epsilon: float = 1e-6,
) -> MGRPOBatch:
    """Create complete-Main and synchronized-Sub training records."""
    if len({rollout.rollout_id for rollout in rollouts}) != len(rollouts):
        raise ValueError("rollout_id values must be unique")

    main_breakdowns = {
        rollout.rollout_id: main_reward(rollout, reward_weights) for rollout in rollouts
    }
    main_advantages = group_relative_advantages(
        rollouts,
        lambda rollout: main_breakdowns[rollout.rollout_id].total,
        epsilon=epsilon,
    )
    main_records = tuple(
        MainTrainingRecord(
            rollout_id=rollout.rollout_id,
            group_key=rollout.group_key,
            reward=main_breakdowns[rollout.rollout_id].total,
            advantage=main_advantages[rollout.rollout_id],
            breakdown=main_breakdowns[rollout.rollout_id],
        )
        for rollout in rollouts
    )

    by_id = {rollout.rollout_id: rollout for rollout in rollouts}
    aligned = align_sub_invocations(rollouts, target_invocations, seed=seed)
    slots_with_rewards = []
    for aligned_rollout in aligned:
        rollout = by_id[aligned_rollout.rollout_id]
        for slot in aligned_rollout.slots:
            if slot.invocation is None:
                slots_with_rewards.append((slot, 0.0, None))
            else:
                breakdown = sub_invocation_reward(rollout, slot.invocation, reward_weights)
                slots_with_rewards.append((slot, breakdown.total, breakdown))

    sub_advantages = _normalize_sub_rewards(slots_with_rewards, epsilon)
    sub_records = tuple(
        SubTrainingRecord(
            rollout_id=slot.rollout_id,
            group_key=slot.group_key,
            slot_index=slot.slot_index,
            invocation=slot.invocation,
            source_invocation_index=slot.source_invocation_index,
            reward=reward,
            advantage=sub_advantages.get((slot.rollout_id, slot.slot_index), 0.0),
            loss_mask=slot.loss_mask,
            duplicated=slot.duplicated,
            breakdown=breakdown,
        )
        for slot, reward, breakdown in slots_with_rewards
    )
    return MGRPOBatch(
        main_records=main_records,
        sub_records=sub_records,
        target_invocations=target_invocations,
    )
