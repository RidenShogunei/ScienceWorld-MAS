"""Strict pass@1 rollout and reporting for ScienceWorld."""

from __future__ import annotations

from dataclasses import dataclass

from scienceworld_mas.bench import BenchmarkScore, EpisodeScore, compute_benchmark_score
from scienceworld_mas.env import EpisodeListMetadata, EpisodeSpec, ScienceWorldRunner

from .policy import ActionDecision, ActionPolicy, PolicyContext, StepTrace


@dataclass(frozen=True)
class EpisodeTrace:
    spec: EpisodeSpec
    task_description: str
    final_score: float
    steps: tuple[StepTrace, ...]
    format_error_count: int
    environment_done: bool

    @property
    def action_valid_count(self) -> int:
        return sum(1 for step in self.steps if step.action_valid)

    @property
    def action_count(self) -> int:
        return len(self.steps)

    def to_episode_score(self) -> EpisodeScore:
        return EpisodeScore(
            task_name=self.spec.task_name,
            variation_id=self.spec.variation_id,
            score=self.final_score,
            steps=len(self.steps),
            action_valid_count=self.action_valid_count,
            action_count=self.action_count,
            format_error_count=self.format_error_count,
        )


@dataclass(frozen=True)
class EvaluationReport:
    episode_list: EpisodeListMetadata | None
    metrics: BenchmarkScore
    episodes: tuple[EpisodeTrace, ...]

    def to_dict(self) -> dict:
        return {
            "episode_list": None if self.episode_list is None else self.episode_list.__dict__,
            "metrics": self.metrics.to_dict(),
            "episodes": [
                {
                    "task_name": item.spec.task_name,
                    "variation_id": item.spec.variation_id,
                    "split": item.spec.split,
                    "task_description": item.task_description,
                    "final_score": item.final_score,
                    "steps": [
                        {
                            "step_index": step.step_index,
                            "observation": step.observation,
                            "action": step.action,
                            "raw_response": step.raw_response,
                            "format_valid": step.format_valid,
                            "action_valid": step.action_valid,
                            "reward": step.reward,
                            "score": step.score,
                            "done": step.done,
                            "next_observation": step.next_observation,
                        }
                        for step in item.steps
                    ],
                    "format_error_count": item.format_error_count,
                    "environment_done": item.environment_done,
                }
                for item in self.episodes
            ],
        }


def _coerce_decision(value: ActionDecision | str | None) -> ActionDecision:
    if isinstance(value, ActionDecision):
        return value
    if value is None:
        return ActionDecision(action=None, format_valid=False)
    return ActionDecision(action=str(value), raw_response=str(value), format_valid=True)


def run_episode(
    runner: ScienceWorldRunner,
    policy: ActionPolicy,
    spec: EpisodeSpec,
    *,
    step_limit: int,
) -> EpisodeTrace:
    """Run one strict pass@1 episode.

    There is no retry, no best-of-N, and no replacement action after a format
    failure. The final score is the official ScienceWorld score from the single
    rollout.
    """

    observation, task_description, reset_info = runner.reset(spec)
    policy.reset_episode(task_description)
    prepare_episode = getattr(policy, "prepare_episode", None)
    if callable(prepare_episode):
        prepare_episode(
            spec=spec,
            task_description=task_description,
            gold_actions=tuple(runner.gold_actions()),
        )
    history: list[StepTrace] = []
    format_error_count = 0
    environment_done = False
    final_score = float(reset_info.get("score", 0.0))

    for step_index in range(step_limit):
        context = PolicyContext(
            task_description=task_description,
            observation=observation,
            step_index=step_index,
            valid_actions=tuple(runner.valid_actions()),
            history=tuple(history),
        )
        decision = _coerce_decision(policy.act(context))
        if not decision.format_valid or decision.action is None:
            format_error_count += 1
            break

        result = runner.step(decision.action)
        trace = StepTrace(
            step_index=step_index,
            observation=observation,
            action=decision.action,
            raw_response=decision.raw_response,
            format_valid=decision.format_valid,
            action_valid=result.action_valid,
            reward=result.reward,
            score=result.score,
            done=result.done,
            next_observation=result.observation,
        )
        history.append(trace)
        observation = result.observation
        final_score = result.score
        environment_done = result.done
        if result.done:
            break

    return EpisodeTrace(
        spec=spec,
        task_description=task_description,
        final_score=final_score,
        steps=tuple(history),
        format_error_count=format_error_count,
        environment_done=environment_done,
    )


def evaluate_episodes(
    runner: ScienceWorldRunner,
    policy: ActionPolicy,
    specs: list[EpisodeSpec],
    *,
    step_limit: int,
    episode_list: EpisodeListMetadata | None = None,
) -> EvaluationReport:
    episodes = tuple(
        run_episode(runner, policy, spec, step_limit=step_limit)
        for spec in specs
    )
    metrics = compute_benchmark_score(item.to_episode_score() for item in episodes)
    return EvaluationReport(
        episode_list=episode_list,
        metrics=metrics,
        episodes=episodes,
    )
