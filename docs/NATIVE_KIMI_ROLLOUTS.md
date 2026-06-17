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

By default the collector uses action-id grounding. The Sub agent sees a ranked
candidate list such as `A0: look around` and must return only:

```text
[action_id]A<num>[/action_id][subtask_done]true|false[/subtask_done]
```

This is intentionally stricter than asking the model to copy a raw action
string. It fixes the most common native-rollout failure mode where the model
understands the subtask but emits an action that is not exactly executable by
ScienceWorld.

The collector also gives the Sub agent a short recent execution history with
reward, score delta, and whether the observation changed. Repeated zero-gain
actions can be removed from the candidate set, so the Sub agent is pushed to
try a different executable action instead of looping.

Valid actions are ranked so task-like commands (`open`, `go`, `pick up`,
`move`, `activate`, `examine`, etc.) appear before graph-style
`connect`/`disconnect` commands. Unrelated unsafe `focus on ...` actions, such
as focusing on rooms or furniture, are filtered unless they target task-like
substances or containers. Near-miss action snapping is still available in the
legacy raw-action mode, but action-id grounding is preferred.

Useful v2 options:

- `--use-action-ids` / `--no-use-action-ids`: default is action-id grounding.
- `--history-limit 6`: number of recent Sub executions shown back to Kimi.
- `--block-no-progress-repeats`: remove repeated zero-gain actions when
  alternatives exist.
- `--include-graph-actions`: include low-level `connect`/`disconnect` actions
  if a task needs them; default is off.

## Convert To SFT

Convert successful or high-score rollouts into chat SFT:

```powershell
python generate_sft_from_kimi_rollouts.py `
  --input data/kimi_mas_rollouts/rollouts_5.jsonl `
  --output-dir data/kimi_mas_sft_5 `
  --min-final-score 20 `
  --valid-actions-only
```

For pilot data, where a rollout may contain useful early steps before later
failure, use prefix-style filtering:

```powershell
python generate_sft_from_kimi_rollouts.py `
  --input data/kimi_mas_rollouts/boil_native10_prefix_source.jsonl `
  --output-dir data/kimi_mas_sft_boil_native10_clean `
  --keep-local-nonnegative-steps `
  --drop-repeated-actions `
  --valid-actions-only
```

This keeps valid actions that do not reduce environment score and removes
repeated actions within the same rollout.

For action-id grounded rollouts, Sub SFT targets are also action ids by default:

```text
[action_id]A<num>[/action_id][subtask_done]true|false[/subtask_done]
```

The converter resolves the id from each step's saved candidate list. Use
`--no-target-action-id` only for legacy raw-action SFT.

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
