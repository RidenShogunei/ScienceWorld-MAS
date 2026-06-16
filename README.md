# ScienceWorld-MAS

An independent research project for hierarchical Main/Sub agent training in
ScienceWorld. The initial supervised baseline uses the official expert
trajectories released with Multi-Square.

For a clean-machine setup and experiment-recording rules, see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Research Question

Can executable environment feedback improve a hierarchical language-agent
policy beyond its expert-trajectory SFT initialization?

The first controlled architecture is intentionally simple:

```text
Main planner -> one subtask -> shared Sub executor -> ScienceWorld
     ^                                                |
     +---------------- observation/reward ------------+
```

This is a sequential hierarchical MAS baseline. Dynamic spawning and a
variable number of independent Sub agents are later experiments, not claims of
the initial baseline.

## Data

Multi-Square provides two complementary ScienceWorld datasets:

- High level: task and planner state -> next subtask.
- Low level: subtask and observation -> executable action and completion flag.

The raw dataset is downloaded at runtime and is not committed to this repo.

```powershell
python -m pip install -r requirements.txt
python prepare_multisquare.py
python audit_multisquare.py --output artifacts/data_audit.json
python generate_sft_data.py
```

Generated files:

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/manifest.json
```

The converter uses deterministic group-based splitting. Main samples are
grouped by normalized task description; Sub samples are grouped by normalized
subtask. This is stricter than randomly splitting individual trajectory steps.

The upstream High-level file contains eight trajectories with one final
subtask but no corresponding reward/score/done label. The loader does not
invent those labels: it drops only those eight unlabeled transitions and
reports the count in `artifacts/data_audit.json`.

## SFT Baseline

Train independent LoRA adapters for the planner and executor:

```powershell
python sft_trainer.py `
  --base-model Qwen/Qwen2.5-1.5B-Instruct `
  --agents both `
  --epochs 2 `
  --log-every-updates 100 `
  --save-every-updates 1000 `
  --use-4bit
```

Best validation-loss adapters are written to:

```text
artifacts/checkpoints/sft/main_agent/best
artifacts/checkpoints/sft/sub_agent/best
```

Long runs also update `main_agent/latest` or `sub_agent/latest` every 1,000
optimizer steps and print elapsed time plus ETA every 100 steps.

Run held-out offline generation evaluation before installing ScienceWorld:

```powershell
python evaluate_sft.py `
  --agent main `
  --adapter artifacts/checkpoints/sft/main_agent/best

python evaluate_sft.py `
  --agent sub `
  --adapter artifacts/checkpoints/sft/sub_agent/best
```

The offline metrics are strict format validity and normalized exact match.
They are diagnostics, not substitutes for full ScienceWorld episode success.

The initial 128-sample pipeline check is documented in
[`docs/SFT_PILOT_REPORT.md`](docs/SFT_PILOT_REPORT.md).
The first executable environment rollout is documented in
[`docs/ENVIRONMENT_SMOKE_REPORT.md`](docs/ENVIRONMENT_SMOKE_REPORT.md).
The planned hierarchical reinforcement-learning structure is documented in
[`docs/MGRPO_DESIGN.md`](docs/MGRPO_DESIGN.md).
Contract-style Main/Sub communication distillation is documented in
[`docs/CONTRACT_DISTILLATION.md`](docs/CONTRACT_DISTILLATION.md).

## Environment Evaluation

After training, run the actual hierarchical policy in ScienceWorld:

```powershell
python evaluate_environment.py `
  --base-model Qwen/Qwen2.5-1.5B-Instruct `
  --main-adapter artifacts/checkpoints/sft/main_agent/best `
  --sub-adapter artifacts/checkpoints/sft/sub_agent/best `
  --split dev `
  --episodes 10
```

Before training or evaluation on a new machine:

```powershell
python doctor.py --smoke-environment
```

## Current Milestone

The first milestone is data validity, followed by:

1. Train separate Main and Sub LoRA adapters from the converted SFT data.
2. Evaluate action syntax, subtask prediction, and full environment success on
   held-out ScienceWorld variations.
3. Establish SFT, Main-only RL, Sub-only RL, and joint RL comparisons.
4. Add dynamic direct/delegate decisions only after the fixed hierarchy works.

## Tests

```powershell
python -m pytest -q
python -m py_compile scienceworld_data.py prepare_multisquare.py audit_multisquare.py generate_sft_data.py
```

## Upstream

- Dataset: https://huggingface.co/datasets/sangeun-park/Multi-Square
- Code: https://github.com/park-sangeun/Multi-Square
- ScienceWorld: https://github.com/allenai/ScienceWorld

The Multi-Square dataset is published under CC-BY-4.0. Preserve attribution
when redistributing derived data.
