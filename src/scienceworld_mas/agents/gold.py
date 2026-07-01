"""Gold-action policy for environment/evaluator smoke tests."""

from __future__ import annotations

from scienceworld_mas.env import EpisodeSpec
from scienceworld_mas.evaluation import ActionDecision, PolicyContext


class GoldActionPolicy:
    """Replay the official ScienceWorld gold action sequence.

    This is an oracle smoke policy, not a model result. It is useful for
    checking that env/evaluation/scoring are wired correctly.
    """

    def __init__(self) -> None:
        self._actions: tuple[str, ...] = ()
        self._index = 0

    def reset_episode(self, task_description: str) -> None:
        self._actions = ()
        self._index = 0

    def prepare_episode(
        self,
        *,
        spec: EpisodeSpec,
        task_description: str,
        gold_actions: tuple[str, ...],
    ) -> None:
        self._actions = gold_actions
        self._index = 0

    def act(self, context: PolicyContext) -> ActionDecision:
        if self._index >= len(self._actions):
            return ActionDecision(action=None, raw_response="", format_valid=False)
        action = self._actions[self._index]
        self._index += 1
        return ActionDecision(action=action, raw_response=action, format_valid=True)
