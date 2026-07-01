"""Load L1 YAML config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


@dataclass
class RewardConfig:
    expert_match: float = 1.0
    action_valid: float = 0.2
    format_valid: float = 0.1
    format_penalty: float = 1.0
    reward_delta_scale: float = 0.0


@dataclass
class TrainConfig:
    agents: str = "main"  # main | sub | both
    save_dir: str = "artifacts/checkpoints/l1_main_step_rl_smoke"
    group_size: int = 4
    iterations: int = 10
    states_per_iter: int = 32
    seed: int = 123
    lr: float = 1e-5
    sub_lr: float | None = None
    kl_beta: float = 0.0
    clip_low: float = 0.1
    clip_high: float = 0.1
    max_grad_norm: float = 1.0
    max_input_length: int = 6656
    max_completion_tokens: int = 350
    main_max_new_tokens: int = 350
    sub_max_new_tokens: int = 32
    rollout_temperature: float = 0.8
    rollout_top_p: float = 0.9
    rollout_sub_do_sample: bool = False
    use_4bit: bool = True
    invalid_format_advantage: float = -1.0
    sub_invalid_format_advantage: float = -1.0
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.01


@dataclass
class StatesConfig:
    episode_list: str = "artifacts/eval/dev_stratified_k5_seed123.json"
    output: str = "artifacts/l1/decision_states_smoke.json"
    # None = all task types present in episode_list (stratified k per task).
    tasks: list[str] | None = field(
        default_factory=lambda: ["find-plant", "find-living-thing", "power-component"]
    )
    variations_per_task: int = 2
    max_actions: int = 32
    history_limit: int = 4
    chunk_size: int = 3


@dataclass
class L1Config:
    base_model: str = "Qwen/Qwen3.5-9B"
    main_adapter: str = ""
    sub_adapter: str = ""
    states: StatesConfig = field(default_factory=StatesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval: dict[str, Any] = field(default_factory=lambda: {"output_json": "artifacts/l1/eval_step_rl.json"})


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)


def load_config(path: str | Path) -> L1Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = L1Config()
    _merge_dataclass(cfg, raw or {})
    return cfg


def train_namespace(cfg: L1Config) -> SimpleNamespace:
    """Minimal namespace for mgrpo_trainer.train_step."""
    t = cfg.train
    return SimpleNamespace(
        clip_low=t.clip_low,
        clip_high=t.clip_high,
        max_input_length=t.max_input_length,
        max_completion_tokens=t.max_completion_tokens,
        max_grad_norm=t.max_grad_norm,
        lr=t.lr,
        main_lr=t.lr,
        sub_lr=t.sub_lr if t.sub_lr is not None else t.lr,
        beta=t.kl_beta,
        agents=t.agents,
        multi_gpu=False,
    )
