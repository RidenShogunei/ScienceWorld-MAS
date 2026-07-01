"""Policy interface used by strict pass@1 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StepTrace:
    step_index: int
    observation: str
    action: str
    raw_response: str
    format_valid: bool
    action_valid: bool
    reward: float
    score: float
    done: bool
    next_observation: str


@dataclass(frozen=True)
class PolicyContext:
    task_description: str
    observation: str
    step_index: int
    history: tuple[StepTrace, ...]


@dataclass(frozen=True)
class ActionDecision:
    action: str | None
    raw_response: str = ""
    format_valid: bool = True


class ActionPolicy(Protocol):
    """Single-action policy evaluated with strict pass@1.

    Implementations may be hierarchical internally, but the evaluation runner
    observes only one executable ScienceWorld action per environment step.
    """

    def reset_episode(self, task_description: str) -> None:
        ...

    def act(self, context: PolicyContext) -> ActionDecision:
        ...
