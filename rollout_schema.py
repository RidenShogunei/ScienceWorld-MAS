"""Typed hierarchical rollout records for ScienceWorld M-GRPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ActionStep:
    step_index: int
    observation: str
    raw_response: str
    action: str | None
    format_valid: bool
    action_valid: bool
    declared_subtask_done: bool
    environment_reward: float
    score_before: float
    score_after: float
    next_observation: str
    environment_done: bool
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    completion_token_ids: list[int] = field(default_factory=list)
    old_logprobs: list[float] = field(default_factory=list)
    handoff: str = "continue"

    @property
    def score_delta(self) -> float:
        return self.score_after - self.score_before


@dataclass
class SubInvocation:
    invocation_id: str
    parent_main_index: int
    subtask: str
    steps: list[ActionStep] = field(default_factory=list)

    @property
    def score_before(self) -> float:
        return self.steps[0].score_before if self.steps else 0.0

    @property
    def score_after(self) -> float:
        return self.steps[-1].score_after if self.steps else self.score_before

    @property
    def score_delta(self) -> float:
        return self.score_after - self.score_before


@dataclass
class MainDecision:
    decision_index: int
    observation: str
    previous_group_actions: list[str]
    raw_response: str
    subtask: str | None
    format_valid: bool
    score_before: float
    invocation_id: str | None = None
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    completion_token_ids: list[int] = field(default_factory=list)
    old_logprobs: list[float] = field(default_factory=list)


@dataclass
class SystemRollout:
    rollout_id: str
    group_key: str
    task_name: str
    variation_id: int
    split: str
    task_description: str
    policy_version: str
    main_decisions: list[MainDecision] = field(default_factory=list)
    sub_invocations: list[SubInvocation] = field(default_factory=list)
    final_score: float = 0.0
    environment_done: bool = False
    truncated: bool = False

    @property
    def success(self) -> bool:
        return self.final_score >= 100.0

    @property
    def action_steps(self) -> list[ActionStep]:
        return [step for invocation in self.sub_invocations for step in invocation.steps]

    def validate(self) -> None:
        decision_indices = [decision.decision_index for decision in self.main_decisions]
        if decision_indices != list(range(len(self.main_decisions))):
            raise ValueError(f"{self.rollout_id}: Main decision indices must be contiguous")

        invocation_ids = [invocation.invocation_id for invocation in self.sub_invocations]
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ValueError(f"{self.rollout_id}: duplicate invocation ids")

        known_invocations = set(invocation_ids)
        for decision in self.main_decisions:
            if decision.invocation_id is not None and decision.invocation_id not in known_invocations:
                raise ValueError(
                    f"{self.rollout_id}: decision references unknown invocation {decision.invocation_id}"
                )
            if decision.old_logprobs and len(decision.old_logprobs) != len(
                decision.completion_token_ids
            ):
                raise ValueError(
                    f"{self.rollout_id}: Main token ids and old log-probs must align"
                )
        for invocation in self.sub_invocations:
            if not 0 <= invocation.parent_main_index < len(self.main_decisions):
                raise ValueError(
                    f"{self.rollout_id}: invalid parent Main index "
                    f"{invocation.parent_main_index}"
                )
            expected_steps = list(range(len(invocation.steps)))
            actual_steps = [step.step_index for step in invocation.steps]
            if actual_steps != expected_steps:
                raise ValueError(
                    f"{self.rollout_id}/{invocation.invocation_id}: "
                    "action step indices must be contiguous"
                )
            for step in invocation.steps:
                if step.old_logprobs and len(step.old_logprobs) != len(
                    step.completion_token_ids
                ):
                    raise ValueError(
                        f"{self.rollout_id}/{invocation.invocation_id}: "
                        "Sub token ids and old log-probs must align"
                    )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemRollout":
        rollout = cls(
            rollout_id=data["rollout_id"],
            group_key=data["group_key"],
            task_name=data["task_name"],
            variation_id=int(data["variation_id"]),
            split=data["split"],
            task_description=data["task_description"],
            policy_version=data["policy_version"],
            main_decisions=[MainDecision(**item) for item in data.get("main_decisions", [])],
            sub_invocations=[
                SubInvocation(
                    invocation_id=item["invocation_id"],
                    parent_main_index=int(item["parent_main_index"]),
                    subtask=item["subtask"],
                    steps=[ActionStep(**step) for step in item.get("steps", [])],
                )
                for item in data.get("sub_invocations", [])
            ],
            final_score=float(data.get("final_score", 0.0)),
            environment_done=bool(data.get("environment_done", False)),
            truncated=bool(data.get("truncated", False)),
        )
        rollout.validate()
        return rollout


def group_key(task_name: str, variation_id: int, split: str) -> str:
    """Stable key identifying rollouts sampled from the same query/environment state."""
    return f"{split}:{task_name}:{variation_id}"
