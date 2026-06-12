"""Observable Main/Sub reward components for ScienceWorld M-GRPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rollout_schema import SubInvocation, SystemRollout


def _normalized_score(score: float) -> float:
    return max(0.0, min(float(score) / 100.0, 1.0))


def _normalized_text(text: str | None) -> str:
    return " ".join((text or "").lower().split())


@dataclass(frozen=True)
class RewardWeights:
    global_score: float = 0.5
    progress: float = 0.3
    format_validity: float = 0.1
    action_validity: float = 0.1
    no_progress_penalty: float = 0.05
    repetition_penalty: float = 0.05
    premature_done_penalty: float = 0.05
    strict_format_gate: bool = True


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def main_reward(
    rollout: SystemRollout,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    """Reward the complete Main trajectory, not individual Main messages."""
    rollout.validate()
    decisions = rollout.main_decisions
    invocations_by_id = {
        invocation.invocation_id: invocation for invocation in rollout.sub_invocations
    }

    format_rate = (
        sum(decision.format_valid for decision in decisions) / len(decisions)
        if decisions
        else 0.0
    )
    positive_progress = sum(max(invocation.score_delta, 0.0) for invocation in rollout.sub_invocations)
    progress = _normalized_score(positive_progress)
    no_progress = sum(
        decision.invocation_id is not None
        and invocations_by_id[decision.invocation_id].score_delta <= 0
        for decision in decisions
    ) / max(len(decisions), 1)

    normalized_subtasks = [
        _normalized_text(decision.subtask)
        for decision in decisions
        if decision.subtask is not None
    ]
    repeats = len(normalized_subtasks) - len(set(normalized_subtasks))
    repetition_rate = repeats / max(len(normalized_subtasks), 1)

    components = {
        "global_score": weights.global_score * _normalized_score(rollout.final_score),
        "progress": weights.progress * progress,
        "format_validity": weights.format_validity * format_rate,
        "no_progress_penalty": -weights.no_progress_penalty * no_progress,
        "repetition_penalty": -weights.repetition_penalty * repetition_rate,
    }
    total = sum(components.values())
    if weights.strict_format_gate and format_rate < 1.0:
        components["strict_format_gate"] = -total
        total = 0.0
    return RewardBreakdown(total=total, components=components)


def sub_invocation_reward(
    rollout: SystemRollout,
    invocation: SubInvocation,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    """Reward one Sub invocation with global outcome and observable local quality."""
    rollout.validate()
    steps = invocation.steps
    format_rate = sum(step.format_valid for step in steps) / max(len(steps), 1)
    valid_rate = sum(step.action_valid for step in steps) / max(len(steps), 1)
    progress = _normalized_score(max(invocation.score_delta, 0.0))

    action_names = [_normalized_text(step.action) for step in steps if step.action]
    repeats = len(action_names) - len(set(action_names))
    repetition_rate = repeats / max(len(action_names), 1)

    premature_done = 0.0
    for step in steps:
        if (
            step.declared_subtask_done
            and not step.environment_done
            and step.score_delta <= 0
        ):
            premature_done = 1.0
            break

    components = {
        "global_score": weights.global_score * _normalized_score(rollout.final_score),
        "progress": weights.progress * progress,
        "format_validity": weights.format_validity * format_rate,
        "action_validity": weights.action_validity * valid_rate,
        "repetition_penalty": -weights.repetition_penalty * repetition_rate,
        "premature_done_penalty": -weights.premature_done_penalty * premature_done,
    }
    total = sum(components.values())
    if weights.strict_format_gate and format_rate < 1.0:
        components["strict_format_gate"] = -total
        total = 0.0
    return RewardBreakdown(total=total, components=components)
