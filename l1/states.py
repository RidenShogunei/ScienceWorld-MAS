"""Expert-replay decision states for L1 step RL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from collect_expert_minimal_rollouts import build_contract, chunk_gold_actions
from eval_episodes import load_episode_list
from scienceworld_env import EpisodeSpec, ScienceWorldRunner

from l1.config import L1Config, load_config
from l1.protocol import expert_action_id, extract_inventory, rank_candidate_actions


@dataclass
class DecisionState:
    state_id: str
    task_name: str
    variation_id: int
    split: str
    step_index: int
    chunk_index: int
    task: str
    observation: str
    planner_observation: str
    inventory: str
    recent_history: list[dict[str, Any]]
    candidate_actions: list[str]
    expert_action: str
    expert_action_id: int | None
    score_before: float
    gold_contract: dict[str, Any]


def _ensure_expert_in_candidates(candidate_actions: list[str], expert_action: str) -> list[str]:
    if expert_action in candidate_actions:
        return candidate_actions
    return list(candidate_actions) + [expert_action]


def collect_states_for_episode(
    runner: ScienceWorldRunner,
    spec: EpisodeSpec,
    *,
    max_actions: int,
    history_limit: int,
    chunk_size: int,
) -> list[DecisionState]:
    runner.env.load(spec.task_name, spec.variation_id, generateGoldPath=True)
    observation, _ = runner.env.reset()
    observation = str(observation)
    task = str(runner.env.get_task_description())
    gold_actions = runner.gold_actions()
    if gold_actions and gold_actions[0].startswith("ERROR:"):
        raise RuntimeError(gold_actions[0])

    states: list[DecisionState] = []
    done = False
    step_count = 0
    recent_history: list[dict[str, Any]] = []

    for chunk_index, action_chunk in enumerate(chunk_gold_actions(gold_actions, chunk_size)):
        if done:
            break
        planner_observation = observation
        gold = build_contract(task, observation, action_chunk, contract_style="semantic")
        for action in action_chunk:
            if done:
                break
            valid_actions = runner.valid_actions()
            context = gold.to_tagged_json() + "\n" + observation
            candidates = rank_candidate_actions(
                valid_actions,
                context=context,
                max_actions=max_actions,
            )
            candidates = _ensure_expert_in_candidates(candidates, action)
            expert_id = expert_action_id(action, candidates)
            try:
                score_before = float(runner.env.get_score())
            except Exception:
                score_before = 0.0

            states.append(
                DecisionState(
                    state_id=f"{spec.task_name}:{spec.variation_id}:{step_count}",
                    task_name=spec.task_name,
                    variation_id=spec.variation_id,
                    split=spec.split,
                    step_index=step_count,
                    chunk_index=chunk_index,
                    task=task,
                    observation=observation,
                    planner_observation=planner_observation,
                    inventory=extract_inventory(observation),
                    recent_history=list(recent_history[-history_limit:]),
                    candidate_actions=candidates,
                    expert_action=action,
                    expert_action_id=expert_id,
                    score_before=score_before,
                    gold_contract=gold.to_payload(),
                )
            )

            next_observation, _reward, done, info, action_valid = runner.step(action)
            step_count += 1
            recent_history.append(
                {"action": action, "action_valid": action_valid, "subtask_done": False}
            )
            observation = str(next_observation)
            _ = info
    return states


def select_episode_specs(
    specs: list[EpisodeSpec],
    *,
    tasks: list[str] | None,
    variations_per_task: int,
) -> list[EpisodeSpec]:
    """Pick up to K variations per task from a stratified episode list."""
    if variations_per_task <= 0:
        raise ValueError("variations_per_task must be positive")

    task_names = tasks if tasks is not None else sorted({spec.task_name for spec in specs})
    selected: list[EpisodeSpec] = []
    for task in task_names:
        task_specs = [spec for spec in specs if spec.task_name == task]
        if not task_specs:
            raise ValueError(f"episode list has no episodes for task {task!r}")
        selected.extend(task_specs[:variations_per_task])
    return selected


def collect_states(cfg: L1Config) -> list[DecisionState]:
    sc = cfg.states
    runner = ScienceWorldRunner()
    _meta, specs = load_episode_list(sc.episode_list)
    selected = select_episode_specs(
        specs,
        tasks=sc.tasks,
        variations_per_task=sc.variations_per_task,
    )

    all_states: list[DecisionState] = []
    for spec in selected:
        all_states.extend(
            collect_states_for_episode(
                runner,
                spec,
                max_actions=sc.max_actions,
                history_limit=sc.history_limit,
                chunk_size=sc.chunk_size,
            )
        )
    runner.close()
    return all_states


def save_states(states: list[DecisionState], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"count": len(states), "states": [asdict(state) for state in states]}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_states(path: str | Path) -> list[DecisionState]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(DecisionState)}
    rows = []
    for row in payload["states"]:
        rows.append(DecisionState(**{key: row[key] for key in allowed if key in row}))
    return rows


EpisodeStepActions = dict[tuple[str, int], dict[int, str]]


def build_episode_step_actions(states: list[DecisionState]) -> EpisodeStepActions:
    """Map (task, variation) -> {step_index: expert_action} for env replay."""
    from collections import defaultdict

    by_episode: EpisodeStepActions = defaultdict(dict)
    for state in states:
        by_episode[(state.task_name, state.variation_id)][state.step_index] = state.expert_action
    return dict(by_episode)


def replay_prefix_for_state(
    state: DecisionState,
    episode_action_index: EpisodeStepActions,
) -> list[str]:
    step_map = episode_action_index.get((state.task_name, state.variation_id))
    if step_map is None:
        raise KeyError(f"no replay actions for episode {(state.task_name, state.variation_id)}")
    missing = [index for index in range(state.step_index) if index not in step_map]
    if missing:
        raise KeyError(f"missing replay steps {missing[:5]} for {state.state_id}")
    return [step_map[index] for index in range(state.step_index)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="l1/config/smoke.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    states = collect_states(cfg)
    save_states(states, cfg.states.output)
    print(f"[l1.states] wrote {len(states)} states -> {cfg.states.output}")


if __name__ == "__main__":
    main()
