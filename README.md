# ScienceWorld-MAS

An independent research project for hierarchical Main/Sub agent training in
ScienceWorld. The initial supervised baseline uses the official expert
trajectories released with Multi-Square.

## V2 Branch Direction

This branch is a clean v2 restructure. The target is a bench-faithful
ScienceWorld project rather than another layer on top of the historical
contract/MGRPO experiments.

Start here:

- [`docs/BENCH_PROTOCOL.md`](docs/BENCH_PROTOCOL.md): fixed evaluation,
  official reward semantics, and System1/System2 training boundaries.
- [`docs/V2_PROJECT_STRUCTURE.md`](docs/V2_PROJECT_STRUCTURE.md): target
  repository layout and migration rules.
- [`configs/bench/bench_faithful_v2.json`](configs/bench/bench_faithful_v2.json):
  the canonical v2 protocol config.

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
Native Kimi Main/Sub environment rollout collection is documented in
[`docs/NATIVE_KIMI_ROLLOUTS.md`](docs/NATIVE_KIMI_ROLLOUTS.md).
The full evolution from static contract distillation to semantic expert
environment rollouts is documented in
[`docs/CONTRACT_SFT_PIPELINE_EVOLUTION.md`](docs/CONTRACT_SFT_PIPELINE_EVOLUTION.md).

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

## Current Milestone (V7 Expert Subtask Contract)

Active dataset: `data/expert_subtask_contract_sft_v3_simple_minimax_sample1000/`
(expert gold replay + MiniMax contract, minimal protocol).

```bash
# SFT
bash scripts/run_sft_expert_subtask_contract_v3.sh

# Stratified-145 environment eval (greedy fp16)
bash scripts/run_eval_expert_subtask_contract_v3.sh

# Sub-only MGRPO baseline (historical V7 RL result)
bash scripts/run_sub_only_mgrpo_expert_subtask_contract_v3.sh

# MrlX-like joint MGRPO ablation
bash scripts/run_joint_mgrpo_expert_subtask_contract_v3_mrlx_like.sh
```

Checkpoints and training logs stay local under `artifacts/checkpoints/` (gitignored).
Experiment summary: [`artifacts/RECENT_DATA_SFT_REPORT.md`](artifacts/RECENT_DATA_SFT_REPORT.md).  
V7 MGRPO lessons (Sub-only + Joint): [`artifacts/V7_CONTRACT_RL_REPORT.md`](artifacts/V7_CONTRACT_RL_REPORT.md).
MrlX-like adaptation notes: [`docs/MRLX_LIKE_MGRPO_NOTES.md`](docs/MRLX_LIKE_MGRPO_NOTES.md).
Pipeline history: [`docs/CONTRACT_SFT_PIPELINE_EVOLUTION.md`](docs/CONTRACT_SFT_PIPELINE_EVOLUTION.md).

## Repository Hygiene

`main` should stay as the runnable, current research baseline. At the moment
that baseline is V7 expert-subtask contract SFT plus the MrlX-like MGRPO
ablation. Older datasets, smoke checkpoints, one-off cache files, and raw
downloaded corpora are intentionally ignored and should not be recommitted.

Tracked data/artifacts are limited to:

```text
data/expert_subtask_contract_sft_v3_simple_minimax_sample1000/
artifacts/eval/dev_stratified_k5_seed123.json
artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json
artifacts/eval/mgrpo_expert_subtask_contract_v3_v3_iter02_stratified145.json
artifacts/eval/mgrpo_expert_subtask_contract_v3_v3_iter10_stratified145.json
artifacts/RECENT_DATA_SFT_REPORT.md
artifacts/V7_CONTRACT_RL_REPORT.md
```

For new experiment directions, create a branch instead of layering another
version onto `main`:

```bash
git switch -c codex/mgrpo-v8-mrlx-db
```

Keep branch-specific scripts and reports named by the experiment ID. Once an
experiment becomes the new baseline, merge only the minimal runnable code,
canonical dataset manifest, final eval JSON, and summary report back to `main`.
Large checkpoints, raw rollouts, provider caches, and temporary aggregate files
such as `all.jsonl` stay local or go to external storage/LFS only when they are
explicitly part of the reproducible release.

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
