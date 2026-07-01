"""Evaluate Main contract quality on fixed states (E1-style)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mgrpo_trainer import MGRPOPolicy

from l1.config import L1Config, load_config
from l1.rollout import evaluate_main_on_state
from l1.states import build_episode_step_actions, load_states
from scienceworld_env import ScienceWorldRunner


def evaluate(
    cfg: L1Config,
    *,
    main_adapter: str | None = None,
    sub_adapter: str | None = None,
) -> dict:
    states = load_states(cfg.states.output)
    episode_action_index = build_episode_step_actions(states)
    adapter = main_adapter or cfg.main_adapter
    sub = sub_adapter or cfg.sub_adapter
    policy = MGRPOPolicy(
        cfg.base_model,
        str(Path(adapter)),
        str(Path(sub)),
        use_4bit=cfg.train.use_4bit,
    )
    runner = ScienceWorldRunner()
    rows = []
    for state in states:
        outcome = evaluate_main_on_state(
            policy,
            runner,
            state,
            max_input_length=cfg.train.max_input_length,
            main_max_new_tokens=cfg.train.main_max_new_tokens,
            sub_max_new_tokens=cfg.train.sub_max_new_tokens,
            episode_action_index=episode_action_index,
        )
        rows.append(
            {
                "state_id": state.state_id,
                "expert_match": outcome.expert_match,
                "action_valid": outcome.action_valid,
                "format_valid": outcome.format_valid,
                "parse_success": outcome.parse_success,
                "reward_delta": outcome.reward_delta,
            }
        )
    runner.close()

    n = len(rows)
    summary = {
        "count": n,
        "expert_match_rate": sum(r["expert_match"] for r in rows) / max(n, 1),
        "action_valid_rate": sum(r["action_valid"] for r in rows) / max(n, 1),
        "format_valid_rate": sum(r["format_valid"] for r in rows) / max(n, 1),
        "parse_success_rate": sum(r["parse_success"] for r in rows) / max(n, 1),
        "mean_reward_delta": sum(r["reward_delta"] for r in rows) / max(n, 1),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="l1/config/smoke.yaml")
    parser.add_argument("--main-adapter", default=None)
    parser.add_argument("--sub-adapter", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    payload = evaluate(
        cfg,
        main_adapter=args.main_adapter,
        sub_adapter=args.sub_adapter,
    )
    output = Path(args.output_json or cfg.eval.get("output_json", "artifacts/l1/eval_step_rl.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
