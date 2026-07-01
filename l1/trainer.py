"""Single-step GRPO trainer for Main and/or Sub."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from mgrpo_trainer import MGRPOPolicy, configure_adapter_training, train_step

from l1.config import L1Config, load_config, train_namespace
from l1.reward import StepOutcome, compute_step_reward
from l1.rollout import (
    MainCompletion,
    SubCompletion,
    generate_main_completion,
    generate_sub_completion,
    probe_sub_action,
)
from l1.states import (
    DecisionState,
    build_episode_step_actions,
    collect_states,
    load_states,
    save_states,
)


def group_relative_advantages(rewards: list[float], epsilon: float = 1e-6) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std <= epsilon:
        return [0.0 for _ in rewards]
    return [(value - mean) / (std + epsilon) for value in rewards]


def sample_states(states: list[DecisionState], count: int, rng: random.Random) -> list[DecisionState]:
    if count <= 0 or count >= len(states):
        return list(states)
    return rng.sample(states, count)


def should_stop_after_iteration(
    *,
    mean_expert: float,
    best_expert: float,
    iterations_without_improvement: int,
    patience: int,
    min_delta: float,
) -> tuple[float, int, bool, str | None]:
    """Return updated best, no-improve counter, stop flag, and reason."""
    if mean_expert > best_expert + min_delta:
        return mean_expert, 0, False, None
    new_count = iterations_without_improvement + 1
    if patience > 0 and new_count >= patience:
        return best_expert, new_count, True, "patience"
    return best_expert, new_count, False, None


def _append_main_sample(
    samples: list[tuple[list[int], list[float], float, list[dict]]],
    completion: MainCompletion,
    advantage: float,
    *,
    format_valid: bool,
    invalid_advantage: float,
) -> None:
    if not completion.completion_token_ids:
        return
    adv = advantage if format_valid else invalid_advantage
    samples.append(
        (
            completion.completion_token_ids,
            completion.old_logprobs,
            adv,
            completion.prompt_messages,
        )
    )


def _append_sub_sample(
    samples: list[tuple[list[int], list[float], float, list[dict]]],
    completion: SubCompletion,
    advantage: float,
    *,
    parse_success: bool,
    invalid_advantage: float,
) -> None:
    if not completion.completion_token_ids:
        return
    adv = advantage if parse_success else invalid_advantage
    samples.append(
        (
            completion.completion_token_ids,
            completion.old_logprobs,
            adv,
            completion.prompt_messages,
        )
    )


def train_l1(cfg: L1Config, *, main_adapter: str | None = None, start_iteration: int = 1) -> None:
    states_path = Path(cfg.states.output)
    if states_path.exists():
        states = load_states(states_path)
        print(f"[l1] loaded {len(states)} states from {states_path}")
    else:
        print(f"[l1] collecting states -> {states_path}")
        states = collect_states(cfg)
        save_states(states, states_path)

    if not states:
        raise RuntimeError("no decision states available")

    episode_action_index = build_episode_step_actions(states)

    t = cfg.train
    agents = t.agents
    if agents not in {"main", "sub", "both"}:
        raise ValueError(f"unsupported train.agents={agents!r}")

    save_dir = Path(t.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from scienceworld_env import ScienceWorldRunner

    runner = ScienceWorldRunner()
    rng = random.Random(t.seed)

    main_src = str(Path(main_adapter or cfg.main_adapter))
    sub_src = str(Path(cfg.sub_adapter))

    best_expert = -1.0
    best_iteration = 0
    iterations_without_improvement = 0
    history: list[dict] = []
    early_stopped = False
    early_stop_reason: str | None = None

    print(
        f"[l1] agents={agents} main={main_src} sub={sub_src} "
        f"main_sample=True sub_sample={t.rollout_sub_do_sample or agents != 'main'}",
        flush=True,
    )

    for iteration in range(start_iteration, t.iterations + 1):
        print(f"\n[l1] iteration {iteration}/{t.iterations}")
        iter_states = sample_states(states, t.states_per_iter, rng)
        policy = MGRPOPolicy(
            cfg.base_model,
            main_src,
            sub_src,
            use_4bit=t.use_4bit,
        )
        if agents == "sub":
            n = configure_adapter_training(policy.model, "sub")
            print(f"[l1] frozen Main — training Sub only ({n} params)", flush=True)
        elif agents == "main":
            n = configure_adapter_training(policy.model, "main")
            print(f"[l1] frozen Sub — training Main only ({n} params)", flush=True)
        else:
            n = configure_adapter_training(policy.model, None)
            print(f"[l1] joint training — Main + Sub ({n} params)", flush=True)

        main_samples: list[tuple[list[int], list[float], float, list[dict]]] = []
        sub_samples: list[tuple[list[int], list[float], float, list[dict]]] = []
        iter_metrics: list[dict] = []

        total_states = len(iter_states)
        total_rollouts = total_states * t.group_size
        print(
            f"[l1] rollout begin: {total_states} states x G={t.group_size} "
            f"= {total_rollouts} samples",
            flush=True,
        )
        rollout_started = time.monotonic()

        main_do_sample = agents != "sub"
        sub_do_sample = t.rollout_sub_do_sample or agents in {"sub", "both"}

        for state_idx, state in enumerate(iter_states, start=1):
            completions: list[MainCompletion] = []
            sub_completions: list[SubCompletion] = []
            outcomes: list[StepOutcome] = []
            rewards: list[float] = []

            for sample_idx in range(t.group_size):
                completion = generate_main_completion(
                    policy,
                    state,
                    max_input_length=t.max_input_length,
                    max_new_tokens=t.main_max_new_tokens,
                    do_sample=main_do_sample,
                    temperature=t.rollout_temperature,
                    top_p=t.rollout_top_p,
                )
                sub_completion = generate_sub_completion(
                    policy,
                    state,
                    completion.contract_text,
                    max_input_length=t.max_input_length,
                    max_new_tokens=t.sub_max_new_tokens,
                    do_sample=sub_do_sample,
                    temperature=t.rollout_temperature,
                    top_p=t.rollout_top_p,
                )
                outcome = probe_sub_action(
                    runner,
                    state,
                    sub_completion,
                    episode_action_index=episode_action_index,
                )
                merged = StepOutcome(
                    expert_match=outcome.expert_match,
                    action_valid=outcome.action_valid,
                    format_valid=completion.format_valid,
                    reward_delta=outcome.reward_delta,
                    parse_success=outcome.parse_success,
                    selected_action_id=outcome.selected_action_id,
                    selected_action=outcome.selected_action,
                )
                reward = compute_step_reward(merged, cfg.reward)
                completions.append(completion)
                sub_completions.append(sub_completion)
                outcomes.append(merged)
                rewards.append(reward)

                done = (state_idx - 1) * t.group_size + sample_idx + 1
                if done % max(1, t.group_size) == 0 or done == total_rollouts:
                    elapsed = time.monotonic() - rollout_started
                    rate = done / max(elapsed, 1e-6)
                    eta_s = (total_rollouts - done) / max(rate, 1e-6)
                    expert_hits = sum(o.expert_match for o in outcomes)
                    fmt_hits = sum(o.format_valid for o in outcomes)
                    parse_hits = sum(o.parse_success for o in outcomes)
                    print(
                        f"[l1] rollout {done}/{total_rollouts} "
                        f"state {state_idx}/{total_states} "
                        f"id={state.state_id} "
                        f"expert={expert_hits}/{t.group_size} "
                        f"format={fmt_hits}/{t.group_size} "
                        f"sub_parse={parse_hits}/{t.group_size} "
                        f"mean_r={sum(rewards)/len(rewards):.3f} "
                        f"elapsed={elapsed:.0f}s eta={eta_s:.0f}s",
                        flush=True,
                    )

            advantages = group_relative_advantages(rewards)
            for completion, sub_completion, advantage, outcome in zip(
                completions, sub_completions, advantages, outcomes
            ):
                if agents in {"main", "both"}:
                    _append_main_sample(
                        main_samples,
                        completion,
                        advantage,
                        format_valid=completion.format_valid,
                        invalid_advantage=t.invalid_format_advantage,
                    )
                if agents in {"sub", "both"}:
                    _append_sub_sample(
                        sub_samples,
                        sub_completion,
                        advantage,
                        parse_success=sub_completion.parse_success,
                        invalid_advantage=t.sub_invalid_format_advantage,
                    )

            iter_metrics.append(
                {
                    "state_id": state.state_id,
                    "mean_reward": sum(rewards) / len(rewards),
                    "expert_match_rate": sum(o.expert_match for o in outcomes) / len(outcomes),
                    "format_valid_rate": sum(o.format_valid for o in outcomes) / len(outcomes),
                    "sub_parse_rate": sum(o.parse_success for o in outcomes) / len(outcomes),
                }
            )

        rollout_elapsed = time.monotonic() - rollout_started
        print(f"[l1] rollout done in {rollout_elapsed:.0f}s", flush=True)

        args_ns = train_namespace(cfg)
        main_metrics = {"loss": 0.0, "approx_kl": 0.0, "n_samples": 0, "clip_fraction": 0.0}
        sub_metrics = {"loss": 0.0, "approx_kl": 0.0, "n_samples": 0, "clip_fraction": 0.0}

        if agents in {"main", "both"}:
            print(f"[l1] main update begin ({len(main_samples)} samples) ...", flush=True)
            main_metrics = train_step(policy, "main", main_samples, args_ns)
            print(
                f"[l1] main loss={main_metrics['loss']:.4f} kl={main_metrics['approx_kl']:.4f} "
                f"samples={main_metrics['n_samples']} clip={main_metrics['clip_fraction']:.2%}"
            )

        if agents in {"sub", "both"}:
            print(f"[l1] sub update begin ({len(sub_samples)} samples) ...", flush=True)
            sub_metrics = train_step(policy, "sub", sub_samples, args_ns)
            print(
                f"[l1] sub loss={sub_metrics['loss']:.4f} kl={sub_metrics['approx_kl']:.4f} "
                f"samples={sub_metrics['n_samples']} clip={sub_metrics['clip_fraction']:.2%}"
            )

        mean_expert = sum(row["expert_match_rate"] for row in iter_metrics) / max(len(iter_metrics), 1)
        mean_format = sum(row["format_valid_rate"] for row in iter_metrics) / max(len(iter_metrics), 1)
        mean_sub_parse = sum(row["sub_parse_rate"] for row in iter_metrics) / max(len(iter_metrics), 1)
        print(
            f"[l1] iter expert_match={mean_expert:.1%} format_valid={mean_format:.1%} "
            f"sub_parse={mean_sub_parse:.1%}"
        )

        prev_best = best_expert
        best_expert, iterations_without_improvement, should_stop, stop_reason = should_stop_after_iteration(
            mean_expert=mean_expert,
            best_expert=best_expert,
            iterations_without_improvement=iterations_without_improvement,
            patience=t.early_stop_patience,
            min_delta=t.early_stop_min_delta,
        )
        if mean_expert > prev_best + t.early_stop_min_delta:
            best_iteration = iteration

        iter_record = {
            "iteration": iteration,
            "agents": agents,
            "mean_expert_match": mean_expert,
            "mean_format_valid": mean_format,
            "mean_sub_parse_rate": mean_sub_parse,
            "main_update_loss": main_metrics["loss"],
            "main_approx_kl": main_metrics["approx_kl"],
            "sub_update_loss": sub_metrics["loss"],
            "sub_approx_kl": sub_metrics["approx_kl"],
            "best_expert_match": best_expert,
            "best_iteration": best_iteration,
            "iterations_without_improvement": iterations_without_improvement,
        }
        history.append(iter_record)
        (save_dir / "train_history.json").write_text(
            json.dumps(
                {
                    "agents": agents,
                    "early_stopped": early_stopped,
                    "early_stop_reason": early_stop_reason,
                    "best_expert_match": best_expert,
                    "best_iteration": best_iteration,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        ckpt = save_dir / f"iter_{iteration:04d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        policy.model.save_pretrained(ckpt)
        policy.tokenizer.save_pretrained(ckpt)
        main_ckpt = ckpt / "main"
        if not (main_ckpt / "adapter_config.json").exists():
            raise RuntimeError(f"expected Main adapter at {main_ckpt}")
        if agents in {"sub", "both"}:
            sub_ckpt = ckpt / "sub"
            if not (sub_ckpt / "adapter_config.json").exists():
                raise RuntimeError(f"expected Sub adapter at {sub_ckpt}")
            sub_src = str(sub_ckpt)
        (ckpt / "iter_metrics.json").write_text(
            json.dumps(iter_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        main_src = str(main_ckpt)
        del policy

        if should_stop:
            early_stopped = True
            early_stop_reason = stop_reason
            print(
                f"[l1] early stop after iter {iteration}: reason={stop_reason} "
                f"(best expert_match={best_expert:.1%} @ iter {best_iteration}, "
                f"no_improve={iterations_without_improvement})",
                flush=True,
            )
            break

    runner.close()
    if early_stopped:
        print(f"[l1] stopped early -> {save_dir} (best iter {best_iteration}, expert_match={best_expert:.1%})")
    else:
        print(f"[l1] done -> {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="l1/config/smoke.yaml")
    parser.add_argument("--main-adapter", default=None, help="override Main adapter path (resume)")
    parser.add_argument("--sub-adapter", default=None, help="override Sub adapter path (resume)")
    parser.add_argument("--start-iteration", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.sub_adapter:
        cfg.sub_adapter = args.sub_adapter
    train_l1(
        cfg,
        main_adapter=args.main_adapter,
        start_iteration=args.start_iteration,
    )


if __name__ == "__main__":
    main()
