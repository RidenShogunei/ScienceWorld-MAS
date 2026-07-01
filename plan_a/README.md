# Plan A SFT

Main outputs a compact plan:

```json
[plan]{"subgoal":"...","focus_objects":["..."]}[/plan]
```

Sub uses action-id over candidate actions with Subgoal/Focus in the prompt.

## Generate data (reuses Multi-Square expert trajectories)

```bash
export PYTHONPATH=.
python -m plan_a.generate_sft_data \
  --output-dir data/plan_a_sft_smoke \
  --sample-size 500
```

## Train (example)

```bash
python sft_trainer.py \
  --base-model Qwen/Qwen3.5-9B \
  --agents main sub \
  --train-data data/plan_a_sft_smoke/train.jsonl \
  --val-data data/plan_a_sft_smoke/val.jsonl \
  --save-dir artifacts/checkpoints/plan_a_sft_smoke
```

Sub rows use `schema: plan_a_v1` and action-id assistant labels.
