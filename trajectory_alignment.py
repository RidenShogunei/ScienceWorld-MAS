"""Group-relative advantages and M-GRPO Sub-invocation alignment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from rollout_schema import SubInvocation, SystemRollout


@dataclass(frozen=True)
class AlignedSubSlot:
    rollout_id: str
    group_key: str
    slot_index: int
    invocation: SubInvocation | None
    source_invocation_index: int | None
    loss_mask: float
    duplicated: bool


@dataclass(frozen=True)
class AlignedSubRollout:
    rollout_id: str
    group_key: str
    slots: tuple[AlignedSubSlot, ...]


def group_relative_advantages(
    rollouts: list[SystemRollout],
    reward_fn: Callable[[SystemRollout], float],
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Compute a z-scored advantage within each same-query rollout group."""
    grouped: dict[str, list[SystemRollout]] = {}
    for rollout in rollouts:
        rollout.validate()
        grouped.setdefault(rollout.group_key, []).append(rollout)

    advantages = {}
    for key, members in grouped.items():
        rewards = [float(reward_fn(rollout)) for rollout in members]
        if not all(math.isfinite(value) for value in rewards):
            raise ValueError(f"group {key} contains a non-finite reward")
        mean = sum(rewards) / len(rewards)
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        std = math.sqrt(variance)
        for rollout, reward in zip(members, rewards):
            advantages[rollout.rollout_id] = (
                (reward - mean) / (std + epsilon) if std > epsilon else 0.0
            )
    return advantages


def align_sub_invocations(
    rollouts: list[SystemRollout],
    target_invocations: int,
    seed: int = 123,
) -> list[AlignedSubRollout]:
    """Align every system rollout to M Sub invocations using index sampling.

    The environment trajectory is never modified. Duplicated slots point to an
    existing invocation and therefore only affect training-sample weighting.
    Rollouts with no Sub invocation receive M masked slots.
    """
    if target_invocations <= 0:
        raise ValueError("target_invocations must be positive")

    rng = random.Random(seed)
    aligned = []
    for rollout in rollouts:
        rollout.validate()
        count = len(rollout.sub_invocations)
        if count == 0:
            indices: list[int | None] = [None] * target_invocations
        elif count >= target_invocations:
            indices = rng.sample(range(count), target_invocations)
        else:
            indices = list(range(count))
            indices.extend(rng.randrange(count) for _ in range(target_invocations - count))
            rng.shuffle(indices)

        seen: set[int] = set()
        slots = []
        for slot_index, source_index in enumerate(indices):
            duplicated = source_index is not None and source_index in seen
            if source_index is not None:
                seen.add(source_index)
            slots.append(
                AlignedSubSlot(
                    rollout_id=rollout.rollout_id,
                    group_key=rollout.group_key,
                    slot_index=slot_index,
                    invocation=(
                        rollout.sub_invocations[source_index]
                        if source_index is not None
                        else None
                    ),
                    source_invocation_index=source_index,
                    loss_mask=0.0 if source_index is None else 1.0,
                    duplicated=duplicated,
                )
            )
        aligned.append(
            AlignedSubRollout(
                rollout_id=rollout.rollout_id,
                group_key=rollout.group_key,
                slots=tuple(slots),
            )
        )
    return aligned


def flatten_aligned_slots(aligned: list[AlignedSubRollout]) -> list[AlignedSubSlot]:
    return [slot for rollout in aligned for slot in rollout.slots]

