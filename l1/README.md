# L1: Main Step RL (Curriculum Level 1)

Single-step Main-only GRPO with a **frozen action-id Sub**. Each training unit is one
decision state — not a full episode.

## Scope

This directory is self-contained. It does **not** depend on `diagnostics/` or episodic
`mgrpo_trainer.py`. It reuses only:

- `scienceworld_env.py`
- `generate_minimal_contract_sft_data.py`
- `collect_expert_minimal_rollouts.py` (`build_contract`)
- `eval_episodes.py`
- `mgrpo_objective.py` (clipped loss)
- `mgrpo_trainer.py` (`MGRPOPolicy`, `configure_adapter_training`, `train_step`)

## Quick start

```bash
export PYTHONPATH=.
export JAVA_HOME=/path/to/jdk-21
export PATH="$JAVA_HOME/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com

# Collect states (once)
python -m l1.states --config l1/config/smoke.yaml

# Smoke train (Main-only, horizon=1)
bash scripts/l1_smoke.sh

# Eval expert_match on fixed states
bash scripts/l1_eval.sh
```

## Success criterion (smoke)

E1-style **main_contract expert_match** should rise from ~31% toward **>50%** without
Main format collapse.

## Layout

```
l1/
  config/smoke.yaml   # single default config
  protocol.py         # action-id Sub protocol
  states.py           # expert replay state collection
  main_prompt.py      # Main chat messages + contract parse
  reward.py           # step reward
  rollout.py          # one Main→Sub→env.step
  trainer.py          # Main-only step GRPO
  eval.py             # fixed-state metrics
```
