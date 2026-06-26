"""M-GRPO Trainer: Group Relative Policy Optimization for ScienceWorld hierarchical MAS."""

from __future__ import annotations

import os

import torch

# Force CUDA init before other imports corrupt it (sandbox compatibility)
if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
    _ = torch.cuda.device_count()

import argparse
import gc
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.optim import AdamW

from agent_protocol import (
    build_main_user_content,
    build_sub_messages,
    fit_sub_messages_for_inference,
    main_system_prompt,
    parse_main_output,
    parse_sub_output,
    sub_system_prompt,
)
from collect_kimi_mas_rollouts import select_candidate_actions
from contract_schema import CommunicationContract
from generate_minimal_contract_sft_data import MinimalContract
from mgrpo_batch import build_mgrpo_batch
from mgrpo_objective import clipped_policy_loss
from rollout_schema import (
    ActionStep,
    MainDecision,
    SubInvocation,
    SystemRollout,
    group_key,
)
from scienceworld_env import EpisodeSpec, ScienceWorldRunner
from scienceworld_rewards import RewardWeights
from sft_trainer import ensure_torch_set_submodule


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

MAIN_PATTERN = re.compile(r"\[subtask\](.*?)\[/subtask\]", re.DOTALL)
SUB_PATTERN = re.compile(
    r"\[action\](.*?)\[/action\]\s*\[subtask_done\](true|false)\[/subtask_done\]",
    re.DOTALL | re.IGNORECASE,
)
TERMINAL_HANDOFFS = frozenset({"complete", "blocked", "need_replan"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NoContext:
    """A no-op context manager for optionally skipping torch.no_grad()."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

def configure_adapter_training(model, train_adapter: str | None) -> int:
    """Enable gradients only for LoRA weights belonging to `train_adapter`.

    When `train_adapter` is None (joint training), all LoRA params train.
    Returns the number of trainable parameters.
    """
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
            continue
        if train_adapter is None:
            param.requires_grad = True
        else:
            # PEFT multi-adapter names look like ...lora_A.main.weight
            param.requires_grad = f".{train_adapter}." in name
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MGRPOPolicy:
    """Load base model + two LoRA adapters.  Provide log-prob-aware generation."""

    def __init__(
        self, base_model: str, main_adapter: str, sub_adapter: str,
        use_4bit: bool, multi_gpu: bool = False, freeze: bool = False,
        device: str | None = None,
    ) -> None:
        ensure_torch_set_submodule()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"

        self._multi_gpu = multi_gpu and torch.cuda.device_count() > 1 and not use_4bit
        kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": "cuda:0"}
        else:
            kwargs["dtype"] = torch.bfloat16 if torch.cuda.device_count() > 0 else torch.float32
            if self._multi_gpu:
                kwargs["device_map"] = "auto"

        base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        base.config.use_cache = True  # needed for generation; turn off during training

        model = PeftModel.from_pretrained(base, main_adapter, adapter_name="main")
        model.load_adapter(sub_adapter, adapter_name="sub")

        model.eval()
        if torch.cuda.device_count() > 0 and not use_4bit and not self._multi_gpu:
            target = device or "cuda:0"
            model = model.to(target)
            self._device_override = torch.device(target)
        else:
            self._device_override = None
        self.model = model
        self._use_4bit = use_4bit
        self._optimizers: dict[str, AdamW] = {}
        if self._multi_gpu:
            print(
                f"[mgrpo] multi-GPU device_map={getattr(model, 'hf_device_map', 'auto')} "
                f"({torch.cuda.device_count()} visible GPUs)",
                flush=True,
            )

    @property
    def device(self):
        if self._device_override is not None:
            return self._device_override
        if self._use_4bit:
            return torch.device("cuda:0")
        return self._input_device()

    def _input_device(self) -> torch.device:
        """Device for input_ids — embed layer on device_map models."""
        if self._use_4bit:
            return torch.device("cuda:0")
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            if hasattr(self.model, "device"):
                return self.model.device
            return next(self.model.parameters()).device

    def generate_with_logprobs(
        self, adapter: str, messages: list[dict],
        max_input_length: int, max_new_tokens: int,
        *,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> tuple[str, list[int], list[float]]:
        """Generate → return (text, token_ids, old_logprobs)."""
        self.model.set_adapter(adapter)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_length,
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p
            gen_kwargs["renormalize_logits"] = True
        if repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        with torch.no_grad():
            gen = self.model.generate(**inputs, **gen_kwargs)
        completion_ids = gen.sequences[0, prompt_len:].tolist()
        scores = torch.stack(gen.scores, dim=0)[:, 0, :]
        lp_matrix = torch.log_softmax(scores, dim=-1)
        old_logprobs = [float(lp_matrix[i, tid]) for i, tid in enumerate(completion_ids)]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return text, completion_ids, old_logprobs

    def logprobs_of_completion(
        self, adapter: str, messages: list[dict],
        completion_ids: list[int], max_input_length: int,
        training: bool = False,
    ):
        """Compute token-level log-probs. Returns list[float] (inference) or
        list[torch.Tensor] (training, with grad preserved)."""
        self.model.set_adapter(adapter)
        if training:
            self.model.train()
            self.model.config.use_cache = False
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=max_input_length,
        )["input_ids"]
        budget = max(max_input_length - len(completion_ids), 0)
        input_ids = prompt_ids[-budget:] + completion_ids
        ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        ctx = _NoContext() if training else torch.no_grad()
        with ctx:
            logits = self.model(input_ids=ids_t).logits[0]  # [seq, vocab]
        logits_for_completion = logits[-len(completion_ids) - 1:-1, :]
        lp = torch.log_softmax(logits_for_completion, dim=-1)
        comp_t = torch.tensor(completion_ids, device=lp.device)
        gathered = lp[torch.arange(len(completion_ids), device=lp.device), comp_t]
        del logits, logits_for_completion, lp

        if not training:
            return [float(v) for v in gathered]
        return [gathered[i] for i in range(len(completion_ids))]


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(
    policy: MGRPOPolicy, runner: ScienceWorldRunner,
    spec: EpisodeSpec, args: argparse.Namespace,
) -> SystemRollout:
    if args.protocol in ("contract", "minimal"):
        return _run_contract_rollout(policy, runner, spec, args)
    return _run_subtask_rollout(policy, runner, spec, args)


def _run_subtask_rollout(
    policy: MGRPOPolicy, runner: ScienceWorldRunner,
    spec: EpisodeSpec, args: argparse.Namespace,
) -> SystemRollout:
    from generate_sft_data import MAIN_SYSTEM, SUB_SYSTEM

    observation, task, _ = runner.reset(spec)
    rollout = SystemRollout(
        rollout_id="", group_key=group_key(spec.task_name, spec.variation_id, spec.split),
        task_name=spec.task_name, variation_id=spec.variation_id, split=spec.split,
        task_description=task, policy_version="sft", final_score=0.0,
    )
    prev_actions: list[str] = []
    step_count, done, inv_cnt = 0, False, 0

    while not done and step_count < args.step_limit and len(rollout.main_decisions) < args.max_subtasks:
        msgs = [
            {"role": "system", "content": MAIN_SYSTEM},
            {"role": "user", "content": f"Task:\n{task}\n\nPlanner state:\n{observation}"},
        ]
        raw, cids, olp = policy.generate_with_logprobs(
            "main", msgs, args.max_input_length, args.main_max_new_tokens,
            do_sample=args.rollout_main_do_sample,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
        )
        m = MAIN_PATTERN.search(raw)
        subtask = m.group(1).strip() if m else None
        dec = MainDecision(
            decision_index=len(rollout.main_decisions),
            observation=observation, previous_group_actions=list(prev_actions),
            raw_response=raw, subtask=subtask, format_valid=m is not None,
            score_before=0.0, prompt_messages=msgs,
            completion_token_ids=cids, old_logprobs=olp,
        )
        rollout.main_decisions.append(dec)
        if subtask is None:
            break

        inv = SubInvocation(
            invocation_id=str(inv_cnt), parent_main_index=dec.decision_index, subtask=subtask,
        )
        inv_cnt += 1
        dec.invocation_id = inv.invocation_id
        actions_done, subtask_done = [], False

        while not done and not subtask_done and step_count < args.step_limit:
            sub_msgs = [
                {"role": "system", "content": SUB_SYSTEM},
                {"role": "user", "content": f"Subtask:\n{subtask}\n\nObservation:\n{observation}"},
            ]
            raw, cids, olp = policy.generate_with_logprobs(
                "sub", sub_msgs, args.max_input_length, args.sub_max_new_tokens,
                do_sample=args.rollout_sub_do_sample,
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
            )
            m = SUB_PATTERN.search(raw)
            act = m.group(1).strip() if m else None
            dd = m.group(2).lower() == "true" if m else False

            try:
                sb = float(runner.env.get_score())
            except Exception:
                sb = rollout.final_score

            if act is not None:
                nobs, rew, done, info, av = runner.step(act)
            else:
                nobs, rew, done, info, av = observation, 0.0, False, {}, False
            step_count += 1
            sa = float(info.get("score", sb))
            rollout.final_score = sa

            inv.steps.append(ActionStep(
                step_index=len(inv.steps), observation=observation,
                raw_response=raw, action=act, format_valid=m is not None,
                action_valid=av, declared_subtask_done=dd,
                environment_reward=float(rew), score_before=sb, score_after=sa,
                next_observation=nobs, environment_done=done,
                prompt_messages=sub_msgs, completion_token_ids=cids, old_logprobs=olp,
            ))
            actions_done.append(act or "")
            observation, subtask_done = nobs, dd

        prev_actions = actions_done
        rollout.sub_invocations.append(inv)

    rollout.truncated = step_count >= args.step_limit
    rollout.environment_done = done
    rollout.validate()
    return rollout


def _run_contract_rollout(
    policy: MGRPOPolicy, runner: ScienceWorldRunner,
    spec: EpisodeSpec, args: argparse.Namespace,
) -> SystemRollout:
    observation, task, _ = runner.reset(spec)
    rollout = SystemRollout(
        rollout_id="", group_key=group_key(spec.task_name, spec.variation_id, spec.split),
        task_name=spec.task_name, variation_id=spec.variation_id, split=spec.split,
        task_description=task, policy_version="minimal" if args.protocol == "minimal" else "contract-sft", final_score=0.0,
    )
    prev_actions: list[str] = []
    step_count, done, inv_cnt = 0, False, 0
    protocol = args.protocol

    while not done and step_count < args.step_limit and len(rollout.main_decisions) < args.max_subtasks:
        main_msgs = [
            {"role": "system", "content": main_system_prompt(protocol)},
            {
                "role": "user",
                "content": build_main_user_content(task, observation, prev_actions),
            },
        ]
        raw, cids, olp = policy.generate_with_logprobs(
            "main", main_msgs, args.max_input_length, args.main_max_new_tokens,
            do_sample=args.rollout_main_do_sample,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            repetition_penalty=args.rollout_main_repetition_penalty,
        )
        plan = parse_main_output(protocol, raw)
        contract = plan if isinstance(plan, (CommunicationContract, MinimalContract)) else None
        subgoal = contract.subgoal if contract else None
        dec = MainDecision(
            decision_index=len(rollout.main_decisions),
            observation=observation,
            previous_group_actions=list(prev_actions),
            raw_response=raw,
            subtask=subgoal,
            format_valid=contract is not None,
            score_before=0.0,
            prompt_messages=main_msgs,
            completion_token_ids=cids,
            old_logprobs=olp,
        )
        rollout.main_decisions.append(dec)
        if contract is None:
            break

        inv = SubInvocation(
            invocation_id=str(inv_cnt),
            parent_main_index=dec.decision_index,
            subtask=contract.to_tagged_json(),
        )
        inv_cnt += 1
        dec.invocation_id = inv.invocation_id
        actions_done: list[str] = []
        group_done = False
        recent_history: list[dict[str, Any]] = []

        while (
            not done
            and not group_done
            and step_count < args.step_limit
            and len(inv.steps) < args.max_steps_per_contract
        ):
            if protocol == "minimal":
                sub_msgs = build_sub_messages(
                    protocol,
                    task_context=contract,
                    observation=observation,
                )
            else:
                valid_actions = runner.valid_actions()
                ranked_actions = select_candidate_actions(
                    valid_actions,
                    max_actions=args.max_valid_actions,
                    rank_actions=args.rank_valid_actions,
                    context=contract.to_tagged_json() + "\n" + observation,
                )
                sub_msgs = fit_sub_messages_for_inference(
                    policy.tokenizer,
                    protocol,
                    task_context=contract,
                    observation=observation,
                    valid_actions=ranked_actions,
                    recent_history=recent_history[-args.history_limit :],
                    max_input_length=args.max_input_length,
                )
            raw, cids, olp = policy.generate_with_logprobs(
                "sub", sub_msgs, args.max_input_length, args.sub_max_new_tokens,
                do_sample=args.rollout_sub_do_sample,
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
            )
            act, dd, handoff, fmt_ok = parse_sub_output(protocol, raw)

            try:
                sb = float(runner.env.get_score())
            except Exception:
                sb = rollout.final_score

            if act is not None:
                nobs, rew, done, info, av = runner.step(act)
            else:
                nobs, rew, done, info, av = observation, 0.0, False, {}, False
            step_count += 1
            sa = float(info.get("score", sb))
            rollout.final_score = sa

            inv.steps.append(ActionStep(
                step_index=len(inv.steps),
                observation=observation,
                raw_response=raw,
                action=act,
                format_valid=fmt_ok,
                action_valid=av,
                declared_subtask_done=dd,
                environment_reward=float(rew),
                score_before=sb,
                score_after=sa,
                next_observation=nobs,
                environment_done=done,
                prompt_messages=sub_msgs,
                completion_token_ids=cids,
                old_logprobs=olp,
                handoff=handoff,
            ))
            actions_done.append(act or "")
            recent_history.append(
                {
                    "action": act,
                    "action_valid": av,
                    "reward": float(rew),
                    "handoff": handoff,
                }
            )
            observation = nobs
            group_done = dd or handoff in TERMINAL_HANDOFFS

        prev_actions = actions_done
        rollout.sub_invocations.append(inv)

    rollout.truncated = step_count >= args.step_limit
    rollout.environment_done = done
    rollout.validate()
    return rollout


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def collect_main_training_samples(
    rollouts: list[SystemRollout],
    rollout_advantages: dict[str, float],
    *,
    invalid_format_advantage: float = -1.0,
) -> list[tuple[list[int], list[float], float, list[dict]]]:
    """Collect Main samples with per-decision advantages (format failures penalized)."""
    samples: list[tuple[list[int], list[float], float, list[dict]]] = []
    for rollout in rollouts:
        rollout_adv = rollout_advantages[rollout.rollout_id]
        for dec in rollout.main_decisions:
            if dec.completion_token_ids and dec.old_logprobs:
                advantage = rollout_adv if dec.format_valid else invalid_format_advantage
                samples.append((
                    dec.completion_token_ids.copy(), dec.old_logprobs.copy(),
                    advantage, dec.prompt_messages,
                ))
    return samples


def build_reward_weights(args: argparse.Namespace) -> RewardWeights:
    format_validity = (
        args.main_format_validity
        if args.agents == "main" and args.main_format_validity is not None
        else args.format_validity
    )
    return RewardWeights(
        global_score=args.reward_global_score,
        progress=args.reward_progress,
        format_validity=format_validity,
        action_validity=args.reward_action_validity
        if args.reward_action_validity is not None
        else 0.1,
        no_progress_penalty=args.reward_no_progress_penalty,
        repetition_penalty=args.reward_repetition_penalty,
        premature_done_penalty=args.reward_premature_done_penalty,
        first_decision_format_penalty=args.main_first_decision_format_penalty,
        strict_format_gate=args.strict_format_gate,
    )


def adapter_learning_rate(args: argparse.Namespace, adapter: str) -> float:
    if adapter == "main" and args.main_lr is not None:
        return args.main_lr
    if adapter == "sub" and args.sub_lr is not None:
        return args.sub_lr
    return args.lr


def collect_sub_training_samples(
    sub_records,
) -> list[tuple[list[int], list[float], float, list[dict]]]:
    """Collect per-invocation Sub samples with slot-level advantages."""
    samples: list[tuple[list[int], list[float], float, list[dict]]] = []
    for record in sub_records:
        if record.loss_mask <= 0 or record.invocation is None:
            continue
        for step in record.invocation.steps:
            if step.completion_token_ids and step.old_logprobs:
                samples.append((
                    step.completion_token_ids.copy(), step.old_logprobs.copy(),
                    record.advantage, step.prompt_messages,
                ))
    return samples


def train_step(
    policy: MGRPOPolicy, adapter: str,
    samples: list[tuple[list[int], list[float], float, list[dict]]],
    args: argparse.Namespace,
) -> dict[str, float]:
    """One GRPO policy update.  Processes completions one-at-a-time
    with backward-after-each to keep GPU memory low."""
    policy.model.train()
    policy.model.set_adapter(adapter)
    policy.model.config.use_cache = False
    need_gc = (
        getattr(policy, "_multi_gpu", False)
        or args.max_input_length >= 4096
        or args.agents == "both"
    )
    if need_gc and not getattr(policy, "_gc_enabled", False):
        try:
            policy.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            policy._gc_enabled = True
            print(f"  [mgrpo] gradient checkpointing enabled ({adapter} update)", flush=True)
        except Exception as exc:
            print(f"  [warn] gradient checkpointing unavailable: {exc}", flush=True)

    # Lazy-init AdamW optimizer per adapter
    if adapter not in policy._optimizers:
        trainable = [p for p in policy.model.parameters() if p.requires_grad]
        lr = adapter_learning_rate(args, adapter)
        policy._optimizers[adapter] = AdamW(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        print(f"[mgrpo] Initialized AdamW for {adapter} adapter ({len(trainable)} params, lr={lr})")
    optimizer = policy._optimizers[adapter]

    if not samples:
        return {"loss": 0.0, "n_samples": 0, "approx_kl": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0}

    # Log reward statistics for debugging
    adv_values = [s[2] for s in samples]
    nonzero_adv = sum(1 for value in adv_values if abs(value) > 1e-8)
    print(
        f"  [debug] {adapter} samples={len(samples)} "
        f"adv_min={min(adv_values):.4f} adv_max={max(adv_values):.4f} "
        f"adv_mean={sum(adv_values)/len(adv_values):.4f} "
        f"adv_nonzero={nonzero_adv}/{len(adv_values)}"
    )
    if nonzero_adv == 0:
        print(f"  [warn] {adapter} update has zero non-zero advantages; loss will be ~0")

    max_input = args.max_input_length
    max_comp = args.max_completion_tokens
    total_loss = 0.0
    total_kl = 0.0
    total_clip = 0.0
    n = 0

    for cids, olp, adv, msgs in samples:
        m = min(len(cids), max_comp)
        cids, olp = cids[:m], olp[:m]

        # Current logprobs (with grad)
        curr_lp_tensors = policy.logprobs_of_completion(adapter, msgs, cids, max_input, training=True)
        curr_lp_tensors = curr_lp_tensors[:m]
        curr_b = torch.stack(curr_lp_tensors).unsqueeze(0)  # [1, m]
        old_b = torch.tensor([olp], dtype=torch.float32, device=curr_b.device)  # [1, m]
        mask_b = torch.ones(1, m, dtype=torch.float32, device=curr_b.device)
        adv_b = torch.tensor([adv], dtype=torch.float32, device=curr_b.device)

        loss, diag = clipped_policy_loss(
            curr_b, old_b, adv_b, mask_b,
            clip_low=args.clip_low, clip_high=args.clip_high,
        )
        loss.backward()

        total_loss += float(loss.detach())
        total_kl += float(diag["approx_kl"])
        total_clip += float(diag["clip_fraction"])
        n += 1

        del curr_lp_tensors, curr_b, old_b, mask_b, adv_b, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if n == 0:
        return {"loss": 0.0, "n_samples": 0, "approx_kl": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0}

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy.model.parameters() if p.requires_grad],
        args.max_grad_norm,
    )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    policy.model.eval()
    policy.model.config.use_cache = True

    return {
        "loss": total_loss / max(n, 1),
        "approx_kl": total_kl / max(n, 1),
        "clip_fraction": total_clip / max(n, 1),
        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def sample_iter_specs(
    specs_pool: list[EpisodeSpec],
    *,
    groups: int,
    group_size: int,
    seed: int,
) -> list[EpisodeSpec]:
    """Sample GRPO groups: `groups` unique queries, each repeated `group_size` times."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not specs_pool:
        return []

    rng = random.Random(seed)
    unique_specs = specs_pool[:]
    rng.shuffle(unique_specs)
    num_groups = min(groups, len(unique_specs))
    if num_groups == 0:
        return []

    selected = unique_specs[:num_groups]
    iter_specs: list[EpisodeSpec] = []
    for spec in selected:
        iter_specs.extend([spec] * group_size)
    rng.shuffle(iter_specs)
    return iter_specs


def _log_group_advantage_stats(batch) -> None:
    from collections import defaultdict

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in batch.main_records:
        grouped[record.group_key].append(record.advantage)
    zero_std_groups = 0
    for key, advantages in grouped.items():
        mean = sum(advantages) / len(advantages)
        variance = sum((value - mean) ** 2 for value in advantages) / len(advantages)
        std = math.sqrt(variance)
        if std <= 1e-6:
            zero_std_groups += 1
        print(
            f"  [debug] group {key}: rollouts={len(advantages)} "
            f"adv_std={std:.4f} adv=[{min(advantages):.3f}, {max(advantages):.3f}]"
        )
    if zero_std_groups:
        print(
            f"  [warn] {zero_std_groups}/{len(grouped)} groups have zero advantage spread; "
            "check rollout sampling and stochastic generation"
        )


def _adapter_sources(args: argparse.Namespace, global_iter: int) -> tuple[str, str]:
    """Resolve Main/Sub adapter dirs for this iteration."""
    if global_iter > 1:
        prev = Path(args.save_dir) / f"iter_{global_iter - 1:04d}"
        main_p, sub_p = prev / "main", prev / "sub"
        if main_p.exists() and sub_p.exists():
            return str(main_p), str(sub_p)
    if args.resume:
        resume_path = Path(args.resume)
        return str(resume_path / "main"), str(resume_path / "sub")
    return args.main_adapter, args.sub_adapter


def _cuda_cleanup() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _rollout_device(args: argparse.Namespace, *, after_train: bool) -> str | None:
    """Use the last GPU for rollout after a sharded train step freed cuda:0."""
    if after_train and args.multi_gpu and torch.cuda.device_count() > 1:
        return f"cuda:{torch.cuda.device_count() - 1}"
    return None


def _release_policy(policy: MGRPOPolicy | None) -> None:
    if policy is None:
        return
    policy._optimizers.clear()
    model = policy.model
    if getattr(policy, "_gc_enabled", False):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    try:
        model.eval()
        if not getattr(policy, "_multi_gpu", False):
            model.cpu()
        else:
            # device_map models: drop accelerate hooks then move shards to CPU.
            try:
                from accelerate.hooks import remove_hook_from_module
                for module in model.modules():
                    if hasattr(module, "_hf_hook"):
                        remove_hook_from_module(module)
            except Exception:
                pass
            try:
                model.cpu()
            except Exception:
                pass
    except Exception:
        pass
    policy.model = None
    del policy.model
    del policy.tokenizer
    del model
    del policy
    for _ in range(3):
        _cuda_cleanup()


def train_mgrpo(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.agents == "sub":
        print("[mgrpo] Sub-only training")
    elif args.agents == "main":
        print("[mgrpo] Main-only training")
    else:
        print("[mgrpo] joint training — Main + Sub")
    if args.multi_gpu:
        print("[mgrpo] multi-GPU train mode: single-GPU rollout → sharded update")

    runner = ScienceWorldRunner(step_limit=args.step_limit)
    specs_pool = _build_spec_pool(runner, args)

    # Determine starting iteration index for resume numbering
    start_iter = 0
    if args.resume:
        try:
            start_iter = int(Path(args.resume).name.split("_")[-1])
        except (ValueError, IndexError):
            pass

    try:
        train_completed = False
        for local_iter in range(1, args.iterations + 1):
            global_iter = start_iter + local_iter
            main_src, sub_src = _adapter_sources(args, global_iter)
            print(f"\n{'='*60}")
            print(f"[mgrpo] iteration {global_iter} (run {local_iter}/{args.iterations})")
            print(f"{'='*60}")
            print(f"[mgrpo] adapters: main={main_src} sub={sub_src}")

            # Sample specs for this iteration: G groups × group_size rollouts each
            iter_specs = sample_iter_specs(
                specs_pool,
                groups=args.groups,
                group_size=args.group_size,
                seed=args.seed + global_iter,
            )

            if len(iter_specs) < args.group_size:
                print(f"[mgrpo] only {len(iter_specs)} specs — skipping")
                continue

            print(
                f"[mgrpo] sampled {len(iter_specs)} rollouts "
                f"({len(iter_specs) // args.group_size} groups × {args.group_size})"
            )

            # ---- Rollout (always single-GPU — device_map breaks sampling) ----
            rollout_multi = False
            rollout_dev = _rollout_device(args, after_train=train_completed)
            print(
                f"[mgrpo] loading rollout policy (multi_gpu={rollout_multi}"
                f"{f', device={rollout_dev}' if rollout_dev else ''}) ...",
                flush=True,
            )
            _cuda_cleanup()
            policy = MGRPOPolicy(
                args.base_model, main_src, sub_src, args.rollout_use_4bit,
                multi_gpu=rollout_multi, device=rollout_dev,
            )
            print(
                f"[mgrpo] rollout quant={'4bit' if args.rollout_use_4bit else 'fp16'} "
                f"main_sample={args.rollout_main_do_sample} sub_sample={args.rollout_sub_do_sample}",
                flush=True,
            )
            print(f"[mgrpo] collecting {len(iter_specs)} rollouts ...")
            rollouts: list[SystemRollout] = []
            for i, spec in enumerate(iter_specs):
                print(f"  [{i+1}/{len(iter_specs)}] {spec.task_name} var={spec.variation_id} ...", end=" ", flush=True)
                rollout = run_rollout(policy, runner, spec, args)
                rollout.rollout_id = f"iter{global_iter:04d}_r{i:04d}"
                rollouts.append(rollout)
                print(f"score={rollout.final_score:.1f} steps={len(rollout.action_steps)}")

            _release_policy(policy)

            # ---- Build batch ----
            rw = build_reward_weights(args)
            batch = build_mgrpo_batch(
                rollouts, args.target_invocations,
                seed=args.seed + global_iter,
                reward_weights=rw,
                epsilon=args.epsilon,
            )
            _log_group_advantage_stats(batch)

            # ---- Train (multi-GPU when requested) ----
            train_multi = args.multi_gpu
            print(f"[mgrpo] loading train policy (multi_gpu={train_multi}) ...")
            _cuda_cleanup()
            policy = MGRPOPolicy(
                args.base_model, main_src, sub_src, args.use_4bit, multi_gpu=train_multi,
            )
            if args.agents == "sub":
                n = configure_adapter_training(policy.model, "sub")
                print(f"[mgrpo] frozen Main — training Sub adapter only ({n} params)")
            elif args.agents == "main":
                n = configure_adapter_training(policy.model, "main")
                print(f"[mgrpo] frozen Sub — training Main adapter only ({n} params)")
            else:
                n = configure_adapter_training(policy.model, None)
                print(f"[mgrpo] joint training — Main + Sub ({n} params)")

            # ---- Train Main ----
            if args.agents in ("main", "both"):
                main_rewards = {r.rollout_id: r.advantage for r in batch.main_records}
                if main_rewards:
                    print(f"\n[mgrpo] Main update ({len(main_rewards)} rollouts) ...")
                    mr_vals = list(main_rewards.values())
                    print(f"  [debug] Main advantages: min={min(mr_vals):.4f} max={max(mr_vals):.4f} mean={sum(mr_vals)/len(mr_vals):.4f}")
                    main_samples = collect_main_training_samples(
                        rollouts,
                        main_rewards,
                        invalid_format_advantage=args.main_invalid_format_advantage,
                    )
                    invalid_n = sum(
                        1
                        for rollout in rollouts
                        for dec in rollout.main_decisions
                        if dec.completion_token_ids and not dec.format_valid
                    )
                    if invalid_n:
                        print(
                            f"  [debug] Main format-invalid decisions={invalid_n} "
                            f"(advantage={args.main_invalid_format_advantage})",
                            flush=True,
                        )
                    metrics = train_step(policy, "main", main_samples, args)
                    print(f"  loss={metrics['loss']:.4f} kl={metrics['approx_kl']:.4f} "
                          f"samples={metrics['n_samples']} clip={metrics['clip_fraction']:.2%}")

            # ---- Train Sub ----
            if args.agents in ("sub", "both"):
                sub_samples = collect_sub_training_samples(batch.sub_records)
                if sub_samples:
                    print(f"[mgrpo] Sub update ({len(sub_samples)} token samples) ...")
                    metrics = train_step(policy, "sub", sub_samples, args)
                    print(f"  loss={metrics['loss']:.4f} kl={metrics['approx_kl']:.4f} "
                          f"samples={metrics['n_samples']} clip={metrics['clip_fraction']:.2%}")
                    if metrics["approx_kl"] > 3.0:
                        print(
                            f"  [warn] Sub approx_kl={metrics['approx_kl']:.2f} is high; "
                            "consider lowering --sub-lr or raising --beta",
                            flush=True,
                        )

            # ---- Save ----
            ckpt = Path(args.save_dir) / f"iter_{global_iter:04d}"
            ckpt.mkdir(parents=True, exist_ok=True)
            policy.model.save_pretrained(ckpt)
            policy.tokenizer.save_pretrained(ckpt)
            (ckpt / "rollouts.json").write_text(
                json.dumps([r.to_dict() for r in rollouts], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[mgrpo] saved → {ckpt}")

            _release_policy(policy)
            train_completed = True

    finally:
        runner.close()


def _build_spec_pool(runner: ScienceWorldRunner, args: argparse.Namespace) -> list[EpisodeSpec]:
    task_names = args.tasks or runner.task_names
    candidates = []
    for tn in task_names:
        for vid in runner.variations(tn, args.split):
            candidates.append(EpisodeSpec(tn, int(vid), args.split))
    return candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--main-adapter", default=None, help="SFT Main adapter (ignored if --resume)")
    p.add_argument("--sub-adapter", default=None, help="SFT Sub adapter (ignored if --resume)")
    p.add_argument("--resume", default=None, help="Resume from M-GRPO checkpoint dir")
    p.add_argument("--agents", choices=("main", "sub", "both"), default="both")
    p.add_argument("--protocol", choices=("subtask", "contract", "minimal"), default="subtask")
    p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--target-invocations", type=int, default=6)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--step-limit", type=int, default=50)
    p.add_argument("--max-subtasks", type=int, default=15)
    p.add_argument("--max-steps-per-contract", type=int, default=6)
    p.add_argument("--max-valid-actions", type=int, default=0)
    p.add_argument(
        "--rank-valid-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--history-limit", type=int, default=6)
    p.add_argument("--max-input-length", type=int, default=768)
    p.add_argument("--max-completion-tokens", type=int, default=64)
    p.add_argument("--main-max-new-tokens", type=int, default=None)
    p.add_argument(
        "--main-repetition-penalty",
        type=float,
        default=None,
        help="Legacy alias; prefer --rollout-main-repetition-penalty for rollouts.",
    )
    p.add_argument(
        "--rollout-main-repetition-penalty",
        type=float,
        default=None,
        help="Main decoding repetition penalty during rollout (default: 1.0). "
        "Values >1 can break stochastic sampling.",
    )
    p.add_argument("--sub-max-new-tokens", type=int, default=64)
    p.add_argument("--rollout-do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rollout-main-do-sample", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rollout-sub-do-sample", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rollout-temperature", type=float, default=0.7)
    p.add_argument("--rollout-top-p", type=float, default=0.9)
    p.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--rollout-use-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Quantization for rollout policy (default: fp16 when --use-4bit and --agents sub).",
    )
    p.add_argument(
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shard model across visible GPUs via device_map=auto (use 2+ GPUs).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--main-lr", type=float, default=None, help="Optional Main adapter learning rate.")
    p.add_argument("--sub-lr", type=float, default=None, help="Optional Sub adapter learning rate.")
    p.add_argument(
        "--strict-format-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero rollout reward when format_validity < 1.0 (recommended for Main RL).",
    )
    p.add_argument("--format-validity", type=float, default=0.1)
    p.add_argument(
        "--main-format-validity",
        type=float,
        default=None,
        help="Main-only override for format_validity reward weight.",
    )
    p.add_argument(
        "--main-first-decision-format-penalty",
        type=float,
        default=0.0,
        help="Extra penalty when the first Main contract fails to parse.",
    )
    p.add_argument(
        "--main-invalid-format-advantage",
        type=float,
        default=-1.0,
        help="Per-decision GRPO advantage for format-invalid Main outputs.",
    )
    p.add_argument("--reward-global-score", type=float, default=0.5)
    p.add_argument("--reward-progress", type=float, default=0.3)
    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument("--clip-high", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.01, help="KL penalty weight against reference (default=0.01)")
    p.add_argument("--reference-model", default=None, help="Path to SFT reference model for KL penalty (defaults to base model)")
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--save-dir", default="artifacts/checkpoints/mgrpo")
    p.add_argument("--reward-action-validity", type=float, default=None,
                   help="Override RewardWeights.action_validity (e.g. 0.3 for Sub-only RL)")
    p.add_argument("--reward-no-progress-penalty", type=float, default=0.05)
    p.add_argument("--reward-repetition-penalty", type=float, default=0.05)
    p.add_argument("--reward-premature-done-penalty", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    contract_like = args.protocol in ("contract", "minimal")
    if args.main_max_new_tokens is None:
        args.main_max_new_tokens = 350 if contract_like else 64
    if args.main_repetition_penalty is None:
        args.main_repetition_penalty = 1.2 if contract_like else 1.0
    if args.rollout_main_repetition_penalty is None:
        # Rollout uses 1.0: repetition_penalty >1 often yields invalid sampling probs.
        args.rollout_main_repetition_penalty = 1.0
    if contract_like and args.max_completion_tokens < 96:
        args.max_completion_tokens = 96
    if args.rollout_use_4bit is None:
        args.rollout_use_4bit = False if (args.use_4bit and args.agents == "sub") else args.use_4bit
    if args.rollout_main_do_sample is None:
        args.rollout_main_do_sample = args.rollout_do_sample if args.agents != "sub" else False
    if args.rollout_sub_do_sample is None:
        args.rollout_sub_do_sample = args.rollout_do_sample
    if not args.resume and not (args.main_adapter and args.sub_adapter):
        p.error("--main-adapter and --sub-adapter required (or use --resume)")
    return args


def main() -> None:
    train_mgrpo(parse_args())


if __name__ == "__main__":
    main()
