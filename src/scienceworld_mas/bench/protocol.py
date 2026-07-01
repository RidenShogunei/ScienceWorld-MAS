"""Stable protocol choices for bench-faithful ScienceWorld experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AgentRole(StrEnum):
    """Role names aligned with the Multi-Square/System1-System2 framing."""

    SYSTEM1_PLANNER = "system1_planner"
    SYSTEM2_EXECUTOR = "system2_executor"


class TrainingStage(StrEnum):
    """Allowed training stages in the v2 baseline."""

    SYSTEM1_SFT = "system1_sft"
    SYSTEM2_BC = "system2_bc"
    SYSTEM2_OFFLINE_RL = "system2_offline_rl"
    SYSTEM2_ONLINE_RL = "system2_online_rl"
    JOINT_RL_ABLATION = "joint_rl_ablation"


class RolloutMode(StrEnum):
    """Evaluation rollout policy."""

    STRICT_PASS_AT_1 = "strict_pass_at_1"


@dataclass(frozen=True)
class BenchProtocol:
    """Experiment contract for the cleaned-up v2 branch.

    The default protocol intentionally keeps Main/System1 stable under SFT and
    treats Main+Sub joint RL as an ablation, not the primary benchmark route.
    """

    name: str
    split: str
    episode_list: str
    metric_primary: str
    comparable_metrics: tuple[str, ...]
    diagnostic_metrics: tuple[str, ...]
    rollout_mode: RolloutMode
    attempts_per_episode: int
    allow_best_of_n: bool
    allow_retry_on_failure: bool
    system1_train_stages: tuple[TrainingStage, ...]
    system2_train_stages: tuple[TrainingStage, ...]
    ablation_stages: tuple[TrainingStage, ...]
    use_official_score_for_rl: bool = True
    preserve_negative_scores: bool = True

    def validate(self) -> None:
        if self.metric_primary != "official_mean_score":
            raise ValueError("v2 bench protocol primary metric must be official_mean_score")
        if self.comparable_metrics != ("official_mean_score",):
            raise ValueError("ScienceWorld comparison metrics should only include official_mean_score")
        if self.rollout_mode != RolloutMode.STRICT_PASS_AT_1:
            raise ValueError("comparable evaluation must use strict pass@1 rollout")
        if self.attempts_per_episode != 1:
            raise ValueError("strict pass@1 requires exactly one attempt per episode")
        if self.allow_best_of_n:
            raise ValueError("best-of-N is not comparable with strict pass@1")
        if self.allow_retry_on_failure:
            raise ValueError("retry-on-failure is not comparable with strict pass@1")
        if not self.use_official_score_for_rl:
            raise ValueError("bench-faithful protocol should use official ScienceWorld score")
        if not self.preserve_negative_scores:
            raise ValueError("negative ScienceWorld scores must not be clamped away")
        if TrainingStage.JOINT_RL_ABLATION in self.system1_train_stages:
            raise ValueError("joint RL belongs in ablation_stages, not the System1 baseline")


DEFAULT_PROTOCOL = BenchProtocol(
    name="scienceworld_mas_v2_bench_faithful",
    split="dev",
    episode_list="artifacts/eval/dev_stratified_k5_seed123.json",
    metric_primary="official_mean_score",
    comparable_metrics=("official_mean_score",),
    diagnostic_metrics=(
        "success_rate",
        "official_score_by_task",
        "action_valid_rate",
        "format_error_rate",
        "negative_score_rate",
        "mean_steps",
    ),
    rollout_mode=RolloutMode.STRICT_PASS_AT_1,
    attempts_per_episode=1,
    allow_best_of_n=False,
    allow_retry_on_failure=False,
    system1_train_stages=(TrainingStage.SYSTEM1_SFT,),
    system2_train_stages=(
        TrainingStage.SYSTEM2_BC,
        TrainingStage.SYSTEM2_OFFLINE_RL,
        TrainingStage.SYSTEM2_ONLINE_RL,
    ),
    ablation_stages=(TrainingStage.JOINT_RL_ABLATION,),
)


DEFAULT_PROTOCOL.validate()
