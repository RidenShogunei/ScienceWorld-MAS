"""M-GRPO Trainer: Group Relative Policy Optimization for ScienceWorld hierarchical MAS."""

from __future__ import annotations

import os

import torch

# Force CUDA init before other imports corrupt it (sandbox compatibility)
if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
    _ = torch.cuda.device_count()

import argparse
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

from generate_sft_data import MAIN_SYSTEM, SUB_SYSTEM
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

class MGRPOPolicy:
    """Load base model + two LoRA adapters.  Provide log-prob-aware generation."""

    def __init__(
        self, base_model: str, main_adapter: str, sub_adapter: str,
        use_4bit: bool, freeze: bool = False,
    ) -> None:
        ensure_torch_set_submodule()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"

        kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = torch.bfloat16 if torch.cuda.device_count() > 0 else torch.float32

        base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        base.config.use_cache = True  # needed for generation; turn off during training

        model = PeftModel.from_pretrained(base, main_adapter, adapter_name="main")
        model.load_adapter(sub_adapter, adapter_name="sub")

        for n, p in model.named_parameters():
            p.requires_grad = "lora" in n.lower()
        if freeze:
            for p in model.parameters():
                p.requires_grad = False

        model.eval()
        if torch.cuda.device_count() > 0 and not use_4bit:
            model = model.to("cuda:0")
        self.model = model
        self._use_4bit = use_4bit
        self._optimizers: dict[str, AdamW] = {}

    @property
    def device(self):
        return next(self.model.parameters()).device if not self._use_4bit else torch.device("cuda:0")

    def generate_with_logprobs(
        self, adapter: str, messages: list[dict],
        max_input_length: int, max_new_tokens: int,
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

        with torch.no_grad():
            gen = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
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
        comp_t = torch.tensor(completion_ids, device=self.device)
        gathered = lp[torch.arange(len(completion_ids), device=self.device), comp_t]

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
    observation, task, _ = runner.reset(spec)
    rollout = SystemRollout(
        rollout_id="", group_key=group_key(spec.task_name, spec.variation_id, spec.split),
        task_name=spec.task_name, variation_id=spec.variation_id, split=spec.split,
        task_description=task, policy_version="sft", final_score=0.0,
    )
    prev_actions: list[str] = []
    step_count, done, inv_cnt = 0, False, 0

    while not done and step_count < args.step_limit and len(rollout.main_decisions) < args.max_subtasks:
        # --- Main ---
        msgs = [
            {"role": "system", "content": MAIN_SYSTEM},
            {"role": "user", "content": f"Task:\n{task}\n\nPlanner state:\n{observation}"},
        ]
        raw, cids, olp = policy.generate_with_logprobs(
            "main", msgs, args.max_input_length, args.main_max_new_tokens,
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

        # --- Sub invocation ---
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


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    policy: MGRPOPolicy, adapter: str,
    rollouts: list[SystemRollout], rewards: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    """One GRPO policy update.  Processes completions one-at-a-time
    with backward-after-each to keep GPU memory low."""
    policy.model.train()
    policy.model.set_adapter(adapter)
    policy.model.config.use_cache = False

    # Lazy-init AdamW optimizer per adapter
    if adapter not in policy._optimizers:
        trainable = [p for p in policy.model.parameters() if p.requires_grad]
        policy._optimizers[adapter] = AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        print(f"[mgrpo] Initialized AdamW for {adapter} adapter ({len(trainable)} params, lr={args.lr})")
    optimizer = policy._optimizers[adapter]

    # Collect samples
    samples: list[tuple[list[int], list[float], float, list[dict]]] = []
    for rollout in rollouts:
        advantage = rewards[rollout.rollout_id]
        if adapter == "main":
            for dec in rollout.main_decisions:
                if dec.completion_token_ids and dec.old_logprobs:
                    samples.append((
                        dec.completion_token_ids.copy(), dec.old_logprobs.copy(),
                        advantage, dec.prompt_messages,
                    ))
        else:
            for inv in rollout.sub_invocations:
                for step in inv.steps:
                    if step.completion_token_ids and step.old_logprobs:
                        samples.append((
                            step.completion_token_ids.copy(), step.old_logprobs.copy(),
                            advantage, step.prompt_messages,
                        ))

    if not samples:
        return {"loss": 0.0, "n_samples": 0}

    # Log reward statistics for debugging
    adv_values = [s[2] for s in samples]
    print(f"  [debug] {adapter} samples={len(samples)} adv_min={min(adv_values):.4f} adv_max={max(adv_values):.4f} adv_mean={sum(adv_values)/len(adv_values):.4f}")

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

    if n == 0:
        return {"loss": 0.0, "n_samples": 0}

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

def train_mgrpo(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[mgrpo] loading policy ...")
    if args.resume:
        resume_path = Path(args.resume)
        main_src = str(resume_path / "main")
        sub_src = str(resume_path / "sub")
        print(f"[mgrpo] resuming from {args.resume}")
    else:
        main_src = args.main_adapter
        sub_src = args.sub_adapter
    policy = MGRPOPolicy(args.base_model, main_src, sub_src, args.use_4bit)

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
        for local_iter in range(1, args.iterations + 1):
            global_iter = start_iter + local_iter
            print(f"\n{'='*60}")
            print(f"[mgrpo] iteration {global_iter} (run {local_iter}/{args.iterations})")
            print(f"{'='*60}")

            # Sample specs for this iteration
            rng = random.Random(args.seed + global_iter)
            g = min(args.groups, len(specs_pool) // args.group_size)
            candidates = specs_pool[:]
            rng.shuffle(candidates)
            iter_specs = []
            groups_seen: set[str] = set()
            for s in candidates:
                gk = group_key(s.task_name, s.variation_id, s.split)
                if gk not in groups_seen:
                    groups_seen.add(gk)
                    iter_specs.append(s)
                    if len(iter_specs) >= g * args.group_size:
                        break

            if len(iter_specs) < args.group_size:
                print(f"[mgrpo] only {len(iter_specs)} specs — skipping")
                continue

            # ---- Rollout ----
            print(f"[mgrpo] collecting {len(iter_specs)} rollouts ...")
            rollouts: list[SystemRollout] = []
            for i, spec in enumerate(iter_specs):
                print(f"  [{i+1}/{len(iter_specs)}] {spec.task_name} var={spec.variation_id} ...", end=" ", flush=True)
                rollout = run_rollout(policy, runner, spec, args)
                rollout.rollout_id = f"iter{global_iter:04d}_r{i:04d}"
                rollouts.append(rollout)
                print(f"score={rollout.final_score:.1f} steps={len(rollout.action_steps)}")

            # ---- Build batch ----
            batch = build_mgrpo_batch(
                rollouts, args.target_invocations,
                seed=args.seed + global_iter,
                reward_weights=RewardWeights(),
                epsilon=args.epsilon,
            )

    # ---- Train Main ----
            if args.agents in ("main", "both"):
                main_rewards = {r.rollout_id: r.advantage for r in batch.main_records}
                if main_rewards:
                    print(f"\n[mgrpo] Main update ({len(main_rewards)} rollouts) ...")
                    # Debug: print reward stats
                    mr_vals = list(main_rewards.values())
                    print(f"  [debug] Main rewards: min={min(mr_vals):.4f} max={max(mr_vals):.4f} mean={sum(mr_vals)/len(mr_vals):.4f}")
                    metrics = train_step(policy, "main", rollouts, main_rewards, args)
                    print(f"  loss={metrics['loss']:.4f} kl={metrics['approx_kl']:.4f} "
                          f"samples={metrics['n_samples']} clip={metrics['clip_fraction']:.2%}")

            # ---- Train Sub ----
            if args.agents in ("sub", "both"):
                sub_rewards: dict[str, list[tuple[float, bool]]] = {}
                for r in batch.sub_records:
                    sub_rewards.setdefault(r.rollout_id, []).append((r.advantage, r.loss_mask > 0))
                sub_avg = {
                    rid: sum(a for a, _ in items) / max(sum(1 for _, m in items if m), 1)
                    for rid, items in sub_rewards.items()
                }
                if sub_avg:
                    print(f"[mgrpo] Sub update ({len(sub_avg)} rollouts) ...")
                    # Debug: print reward stats
                    sr_vals = list(sub_avg.values())
                    print(f"  [debug] Sub rewards: min={min(sr_vals):.4f} max={max(sr_vals):.4f} mean={sum(sr_vals)/len(sr_vals):.4f}")
                    metrics = train_step(policy, "sub", rollouts, sub_avg, args)
                    print(f"  loss={metrics['loss']:.4f} kl={metrics['approx_kl']:.4f} "
                          f"samples={metrics['n_samples']} clip={metrics['clip_fraction']:.2%}")

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
    p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--target-invocations", type=int, default=6)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--step-limit", type=int, default=50)
    p.add_argument("--max-subtasks", type=int, default=15)
    p.add_argument("--max-input-length", type=int, default=768)
    p.add_argument("--max-completion-tokens", type=int, default=64)
    p.add_argument("--main-max-new-tokens", type=int, default=64)
    p.add_argument("--sub-max-new-tokens", type=int, default=64)
    p.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument("--clip-high", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.01, help="KL penalty weight against reference (default=0.01)")
    p.add_argument("--reference-model", default=None, help="Path to SFT reference model for KL penalty (defaults to base model)")
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--save-dir", default="artifacts/checkpoints/mgrpo")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    if not args.resume and not (args.main_adapter and args.sub_adapter):
        p.error("--main-adapter and --sub-adapter required (or use --resume)")
    return args


def main() -> None:
    train_mgrpo(parse_args())


if __name__ == "__main__":
    main()
