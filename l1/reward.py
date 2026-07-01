"""Step-level reward for Main contract quality."""

from __future__ import annotations

from dataclasses import dataclass

from l1.config import RewardConfig


@dataclass(frozen=True)
class StepOutcome:
    expert_match: bool
    action_valid: bool
    format_valid: bool
    reward_delta: float
    parse_success: bool
    selected_action_id: int | None
    selected_action: str | None


def compute_step_reward(outcome: StepOutcome, weights: RewardConfig) -> float:
    total = 0.0
    if outcome.expert_match:
        total += weights.expert_match
    if outcome.action_valid:
        total += weights.action_valid
    if outcome.format_valid:
        total += weights.format_valid
    if not outcome.format_valid and weights.format_penalty > 0:
        total -= weights.format_penalty
    if weights.reward_delta_scale:
        delta = max(-1.0, min(1.0, outcome.reward_delta / 10.0))
        total += weights.reward_delta_scale * delta
    return total
