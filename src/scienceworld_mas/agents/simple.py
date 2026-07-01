"""Deterministic non-model policies for smoke tests."""

from __future__ import annotations

from scienceworld_mas.evaluation import ActionDecision, PolicyContext


class FirstValidActionPolicy:
    """Always choose the first currently valid environment action.

    This policy is intentionally weak; it exists only to test strict pass@1
    plumbing without loading an LLM.
    """

    def reset_episode(self, task_description: str) -> None:
        return None

    def act(self, context: PolicyContext) -> ActionDecision:
        if not context.valid_actions:
            return ActionDecision(action=None, raw_response="", format_valid=False)
        action = context.valid_actions[0]
        return ActionDecision(action=action, raw_response=action, format_valid=True)
