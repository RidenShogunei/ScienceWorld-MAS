# Native Kimi MAS Rollouts

## Purpose

This pipeline collects Main/Sub trajectories from live ScienceWorld interaction
instead of annotating static expert trajectories.

```text
Kimi Main -> structured contract
Kimi Sub  -> executable action from current observation and valid actions
ScienceWorld -> reward, score, next observation
```

The resulting rollouts are closer to the data-construction style used by
AgentGym-derived role-specific datasets: a strong model interacts with the
environment, then the conversation is converted into Main/Sub SFT instances.

## Collect Rollouts

Set the key only in the shell. Do not commit it.

```powershell
$env:KIMI_CODE_API_KEY = "..."

python collect_kimi_mas_rollouts.py `
  --split train `
  --episodes 5 `
  --step-limit 30 `
  --max-subtasks 10 `
  --output data/kimi_mas_rollouts/rollouts_5.jsonl `
  --report-output artifacts/kimi_mas_rollouts/report_5.json
```

The collector provides the current valid ScienceWorld actions to the Sub agent
and can perform one repair call when the Sub output is malformed or not in the
valid action set.
Valid actions are ranked so task-like commands (`open`, `go`, `pick up`,
`move`, `activate`, `examine`, etc.) appear before graph-style
`connect`/`disconnect` commands. Near-miss actions can be snapped to a valid
action, but task-like actions are not snapped into graph-style commands.

## Convert To SFT

Convert successful or high-score rollouts into chat SFT:

```powershell
python generate_sft_from_kimi_rollouts.py `
  --input data/kimi_mas_rollouts/rollouts_5.jsonl `
  --output-dir data/kimi_mas_sft_5 `
  --min-final-score 20 `
  --valid-actions-only
```

Output:

```text
data/kimi_mas_sft_5/train.jsonl
data/kimi_mas_sft_5/val.jsonl
data/kimi_mas_sft_5/test.jsonl
data/kimi_mas_sft_5/manifest.json
```

## Comparison Baselines

Use three data sources separately:

- Original Multi-Square SFT: static high/low expert trajectories.
- Contract annotation SFT: static trajectories plus Kimi communication labels.
- Native Kimi MAS SFT: Kimi-generated Main/Sub interaction trajectories.

This separation makes the central ablation clear: whether native communication
and environment feedback during data generation help more than post-hoc
annotation of existing expert trajectories.
