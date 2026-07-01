from dataclasses import dataclass

from scienceworld_mas.env import EpisodeSpec, StepResult
from scienceworld_mas.evaluation import ActionDecision, PolicyContext, run_episode


class SequencePolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def reset_episode(self, task_description: str) -> None:
        self.index = 0

    def act(self, context: PolicyContext) -> ActionDecision:
        if self.index >= len(self.actions):
            return ActionDecision(action=None, format_valid=False)
        action = self.actions[self.index]
        self.index += 1
        return ActionDecision(action=action, raw_response=action, format_valid=True)


@dataclass
class FakeRunner:
    scores: list[float]
    valid_actions: set[str]

    def reset(self, spec):
        self.step_index = 0
        return "start", "do task", {"score": 0.0}

    def step(self, action: str) -> StepResult:
        score = self.scores[self.step_index]
        self.step_index += 1
        return StepResult(
            observation=f"obs {self.step_index}",
            reward=score,
            done=self.step_index >= len(self.scores),
            info={"score": score},
            action_valid=action in self.valid_actions,
        )


def test_run_episode_records_single_pass_official_score():
    runner = FakeRunner(scores=[1.0, -100.0], valid_actions={"look", "bad"})
    policy = SequencePolicy(["look", "bad"])
    trace = run_episode(
        runner,
        policy,
        EpisodeSpec("task", 0, "dev"),
        step_limit=50,
    )
    assert trace.final_score == -100.0
    assert trace.environment_done
    assert trace.action_count == 2
    assert trace.to_episode_score().score == -100.0


def test_run_episode_stops_on_format_error_without_retry():
    runner = FakeRunner(scores=[100.0], valid_actions={"finish"})
    policy = SequencePolicy([])
    trace = run_episode(
        runner,
        policy,
        EpisodeSpec("task", 0, "dev"),
        step_limit=50,
    )
    assert trace.final_score == 0.0
    assert trace.format_error_count == 1
    assert trace.action_count == 0
