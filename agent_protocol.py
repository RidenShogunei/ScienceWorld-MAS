"""Shared Main/Sub output parsing and prompt helpers for eval and rollout."""

from __future__ import annotations

import json
import re
from typing import Any

from collect_kimi_mas_rollouts import (
    parse_contract_response,
    parse_minimal_contract_response,
    parse_sub_response,
)
from contract_schema import CommunicationContract, parse_contract_text
from generate_contract_sft_data import CONTRACT_MAIN_SYSTEM, CONTRACT_SUB_SYSTEM
from generate_minimal_contract_sft_data import (
    MINIMAL_MAIN_SYSTEM,
    MINIMAL_SUB_SYSTEM,
    MinimalContract,
)
from generate_sft_data import MAIN_SYSTEM, SUB_SYSTEM

ContractLike = CommunicationContract | MinimalContract


MAIN_SUBTASK_PATTERN = re.compile(r"\[subtask\](.*?)\[/subtask\]", re.DOTALL)
SUB_SUBTASK_PATTERN = re.compile(
    r"\[action\](.*?)\[/action\]\s*\[subtask_done\](true|false)\[/subtask_done\]",
    re.DOTALL | re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def build_main_user_content(task: str, observation: str, group_actions: list[str]) -> str:
    state = f"Group action:{group_actions}. Current observation: {observation}"
    return f"Task:\n{task}\n\nPlanner state:\n{state}"


def build_sub_user_content(
    protocol: str,
    *,
    task_context: str | ContractLike,
    observation: str,
    valid_actions: list[str] | None = None,
    recent_history: list[dict[str, Any]] | None = None,
) -> str:
    if protocol == "subtask":
        assert isinstance(task_context, str)
        return f"Subtask:\n{task_context}\n\nObservation:\n{observation}"

    assert isinstance(task_context, (CommunicationContract, MinimalContract))
    parts = [
        f"Contract:\n{task_context.to_tagged_json()}",
        f"Observation:\n{observation}",
    ]
    if recent_history:
        parts.append(
            "Recent execution history:\n"
            + json.dumps(recent_history, ensure_ascii=False, indent=2)
        )
    if valid_actions is not None:
        valid_text = "\n".join(f"- {action}" for action in valid_actions)
        parts.append(f"Valid actions:\n{valid_text}")
    return "\n\n".join(parts)


def build_sub_messages(
    protocol: str,
    *,
    task_context: str | ContractLike,
    observation: str,
    valid_actions: list[str] | None = None,
    recent_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": sub_system_prompt(protocol)},
        {
            "role": "user",
            "content": build_sub_user_content(
                protocol,
                task_context=task_context,
                observation=observation,
                valid_actions=valid_actions,
                recent_history=recent_history,
            ),
        },
    ]


def chat_prompt_token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def _truncate_observation(observation: str, *, max_chars: int) -> str:
    if len(observation) <= max_chars:
        return observation
    if max_chars <= 3:
        return observation[:max_chars]
    return observation[: max_chars - 3] + "..."


def infer_protocol_from_schema(schema: str) -> str:
    if schema.startswith("minimal"):
        return "minimal"
    if schema.startswith("native") or schema.startswith("contract"):
        return "contract"
    return "contract"


def parse_env_sub_user(user: str) -> tuple[CommunicationContract, str, list[str], list[dict[str, Any]]]:
    contract = parse_contract_text(user)
    observation = ""
    obs_match = re.search(
        r"Observation:\n(.*?)(?:\n\nRecent execution history:|\n\nValid actions:|\Z)",
        user,
        re.DOTALL,
    )
    if obs_match:
        observation = obs_match.group(1).strip()

    history: list[dict[str, Any]] = []
    hist_match = re.search(
        r"Recent execution history:\n(.*?)(?:\n\nValid actions:|\Z)",
        user,
        re.DOTALL,
    )
    if hist_match:
        history = json.loads(hist_match.group(1).strip())

    valid_actions: list[str] = []
    if "Valid actions:" in user:
        valid_block = user.split("Valid actions:\n", 1)[1]
        valid_actions = [
            line[2:].strip()
            for line in valid_block.splitlines()
            if line.startswith("- ")
        ]
    return contract, observation, valid_actions, history


def fit_sub_messages_for_inference(
    tokenizer,
    protocol: str,
    *,
    task_context: str | ContractLike,
    observation: str,
    valid_actions: list[str] | None,
    recent_history: list[dict[str, Any]] | None,
    max_input_length: int,
) -> list[dict[str, str]]:
    """Build Sub chat messages that fit within max_input_length.

    Contract and observation are preserved; valid actions and history are trimmed
    before observation text is shortened.
    """
    if protocol == "subtask" or max_input_length <= 0:
        return build_sub_messages(
            protocol,
            task_context=task_context,
            observation=observation,
            valid_actions=valid_actions,
            recent_history=recent_history,
        )

    actions = list(valid_actions or [])
    history = list(recent_history or [])
    obs = observation

    def messages_for(
        action_count: int,
        history_count: int,
        observation_text: str,
    ) -> list[dict[str, str]]:
        trimmed_actions = actions[:action_count] if action_count > 0 else None
        trimmed_history = history[-history_count:] if history_count > 0 else None
        return build_sub_messages(
            protocol,
            task_context=task_context,
            observation=observation_text,
            valid_actions=trimmed_actions,
            recent_history=trimmed_history,
        )

    def fits(action_count: int, history_count: int, observation_text: str) -> bool:
        return chat_prompt_token_count(
            tokenizer,
            messages_for(action_count, history_count, observation_text),
        ) <= max_input_length

    action_count = 0
    if actions:
        lo, hi = 0, len(actions)
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(mid, len(history), obs):
                action_count = mid
                lo = mid + 1
            else:
                hi = mid - 1

    history_count = len(history)
    while history_count > 0 and not fits(action_count, history_count, obs):
        history_count -= 1

    if not fits(action_count, history_count, obs):
        lo, hi = 0, len(obs)
        best = min(256, len(obs))
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = _truncate_observation(obs, max_chars=mid)
            if fits(action_count, history_count, candidate):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        obs = _truncate_observation(obs, max_chars=best)

    return messages_for(action_count, history_count, obs)


def main_system_prompt(protocol: str) -> str:
    if protocol == "minimal":
        return MINIMAL_MAIN_SYSTEM
    return CONTRACT_MAIN_SYSTEM if protocol == "contract" else MAIN_SYSTEM


def sub_system_prompt(protocol: str) -> str:
    if protocol == "minimal":
        return MINIMAL_SUB_SYSTEM
    if protocol == "contract":
        return (
            CONTRACT_SUB_SYSTEM
            + " Include [handoff]continue|complete|blocked|need_replan[/handoff]."
        )
    return SUB_SYSTEM


def parse_main_output(protocol: str, text: str) -> ContractLike | str | None:
    if protocol == "minimal":
        return parse_minimal_contract_response(text)
    if protocol == "contract":
        return parse_contract_response(text)
    match = MAIN_SUBTASK_PATTERN.search(text)
    return match.group(1).strip() if match else None


def parse_sub_output(protocol: str, text: str) -> tuple[str | None, bool, str, bool]:
    if protocol in ("contract", "minimal"):
        action, done, handoff, valid = parse_sub_response(text)
        return action, done, handoff, valid
    match = SUB_SUBTASK_PATTERN.search(text)
    if not match:
        return None, False, "continue", False
    done = match.group(2).lower() == "true"
    handoff = "complete" if done else "continue"
    return match.group(1).strip(), done, handoff, True


def main_outputs_equal(
    protocol: str,
    predicted: ContractLike | str | None,
    expected: ContractLike | str | None,
) -> bool:
    if predicted is None or expected is None:
        return False
    if protocol in ("contract", "minimal"):
        assert isinstance(predicted, (CommunicationContract, MinimalContract))
        assert isinstance(expected, (CommunicationContract, MinimalContract))
        if isinstance(predicted, MinimalContract) or isinstance(expected, MinimalContract):
            return predicted.to_tagged_json() == expected.to_tagged_json()
        return predicted.to_payload() == expected.to_payload()
    return normalize_text(str(predicted)) == normalize_text(str(expected))


def sub_outputs_equal(
    protocol: str,
    predicted: tuple[str | None, bool, str, bool],
    expected: tuple[str | None, bool, str, bool],
) -> bool:
    if not predicted[3] or not expected[3]:
        return False
    pred_action, pred_done, pred_handoff, _ = predicted
    exp_action, exp_done, exp_handoff, _ = expected
    if normalize_text(pred_action or "") != normalize_text(exp_action or ""):
        return False
    if pred_done != exp_done:
        return False
    if protocol == "contract" and pred_handoff != exp_handoff:
        return False
    return True
