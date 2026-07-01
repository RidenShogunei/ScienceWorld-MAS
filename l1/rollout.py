"""Single-step Main → Sub → env rollout for L1."""

from __future__ import annotations

from dataclasses import dataclass

from mgrpo_trainer import MGRPOPolicy
from scienceworld_env import ScienceWorldRunner

from l1.main_prompt import contract_text_for_sub, main_messages
from l1.protocol import build_action_id_messages, decode_action_id, parse_action_id_response
from l1.reward import StepOutcome
from l1.states import DecisionState, EpisodeStepActions, replay_prefix_for_state


@dataclass
class MainCompletion:
    raw_contract: str
    contract_text: str
    format_valid: bool
    prompt_messages: list[dict[str, str]]
    completion_token_ids: list[int]
    old_logprobs: list[float]


@dataclass
class SubCompletion:
    raw_response: str
    action_id: int | None
    parse_success: bool
    prompt_messages: list[dict[str, str]]
    completion_token_ids: list[int]
    old_logprobs: list[float]


def replay_to_state(
    runner: ScienceWorldRunner,
    state: DecisionState,
    *,
    episode_action_index: EpisodeStepActions | None = None,
) -> None:
    runner.env.load(state.task_name, state.variation_id, generateGoldPath=True)
    observation, _ = runner.env.reset()
    _ = observation
    if episode_action_index is not None:
        prefix = replay_prefix_for_state(state, episode_action_index)
    else:
        gold = runner.gold_actions()
        prefix = gold[: state.step_index]
    for action in prefix:
        runner.step(action)


def generate_main_completion(
    policy: MGRPOPolicy,
    state: DecisionState,
    *,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> MainCompletion:
    messages = main_messages(state.task, state.planner_observation, [])
    raw, cids, olp = policy.generate_with_logprobs(
        "main",
        messages,
        max_input_length,
        max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    contract_text, format_valid = contract_text_for_sub(raw)
    return MainCompletion(
        raw_contract=raw,
        contract_text=contract_text,
        format_valid=format_valid,
        prompt_messages=messages,
        completion_token_ids=cids,
        old_logprobs=olp,
    )


def generate_sub_completion(
    policy: MGRPOPolicy,
    state: DecisionState,
    contract_text: str,
    *,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> SubCompletion:
    sub_messages = build_action_id_messages(
        task=state.task,
        observation=state.observation,
        candidate_actions=state.candidate_actions,
        recent_history=state.recent_history,
        contract=contract_text,
    )
    raw, cids, olp = policy.generate_with_logprobs(
        "sub",
        sub_messages,
        max_input_length,
        max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    action_id = parse_action_id_response(raw)
    return SubCompletion(
        raw_response=raw,
        action_id=action_id,
        parse_success=action_id is not None,
        prompt_messages=sub_messages,
        completion_token_ids=cids,
        old_logprobs=olp,
    )


def probe_sub_action(
    runner: ScienceWorldRunner,
    state: DecisionState,
    sub_completion: SubCompletion,
    *,
    episode_action_index: EpisodeStepActions | None = None,
) -> StepOutcome:
    replay_to_state(runner, state, episode_action_index=episode_action_index)
    selected_action = decode_action_id(sub_completion.action_id, state.candidate_actions)

    if selected_action is None:
        return StepOutcome(
            expert_match=False,
            action_valid=False,
            format_valid=True,
            reward_delta=0.0,
            parse_success=sub_completion.parse_success,
            selected_action_id=sub_completion.action_id,
            selected_action=None,
        )

    score_before = float(getattr(runner.env, "get_score", lambda: state.score_before)())
    _obs, reward, _done, info, action_valid = runner.step(selected_action)
    score_after = float(info.get("score", score_before))
    expert_match = sub_completion.action_id == state.expert_action_id

    return StepOutcome(
        expert_match=expert_match,
        action_valid=bool(action_valid),
        format_valid=True,
        reward_delta=float(reward),
        parse_success=sub_completion.parse_success,
        selected_action_id=sub_completion.action_id,
        selected_action=selected_action,
    )


def run_sub_and_probe(
    policy: MGRPOPolicy,
    runner: ScienceWorldRunner,
    state: DecisionState,
    contract_text: str,
    *,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    episode_action_index: EpisodeStepActions | None = None,
) -> StepOutcome:
    sub_completion = generate_sub_completion(
        policy,
        state,
        contract_text,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    return probe_sub_action(
        runner,
        state,
        sub_completion,
        episode_action_index=episode_action_index,
    )


def evaluate_main_on_state(
    policy: MGRPOPolicy,
    runner: ScienceWorldRunner,
    state: DecisionState,
    *,
    max_input_length: int,
    main_max_new_tokens: int,
    sub_max_new_tokens: int,
    episode_action_index: EpisodeStepActions | None = None,
) -> StepOutcome:
    completion = generate_main_completion(
        policy,
        state,
        max_input_length=max_input_length,
        max_new_tokens=main_max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    outcome = run_sub_and_probe(
        policy,
        runner,
        state,
        completion.contract_text,
        max_input_length=max_input_length,
        max_new_tokens=sub_max_new_tokens,
        episode_action_index=episode_action_index,
    )
    return StepOutcome(
        expert_match=outcome.expert_match,
        action_valid=outcome.action_valid,
        format_valid=completion.format_valid,
        reward_delta=outcome.reward_delta,
        parse_success=outcome.parse_success,
        selected_action_id=outcome.selected_action_id,
        selected_action=outcome.selected_action,
    )
