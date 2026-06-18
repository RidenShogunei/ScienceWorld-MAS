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

Set the key only in the shell or pass a local key file. Do not commit it. The
collector calls Kimi Code through its Anthropic-compatible HTTP endpoint by
default; it does not require the Kimi CLI.

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

If the key is stored in a local file:

```powershell
python collect_kimi_mas_rollouts.py `
  --api-key-file C:\Users\chenj\Desktop\kimi.txt `
  --split train `
  --episodes 5 `
  --step-limit 30 `
  --max-subtasks 10 `
  --output data/kimi_mas_rollouts/rollouts_5.jsonl `
  --report-output artifacts/kimi_mas_rollouts/report_5.json
```

The default endpoint is `https://api.kimi.com/coding/v1/messages` with model
`kimi-for-coding`.

The collector uses raw action grounding. The Sub agent sees the
environment-provided valid action list and must copy one executable action
exactly:

```text
[action]look around[/action][subtask_done]true|false[/subtask_done][handoff]continue|complete|blocked|need_replan[/handoff]
```

Because the prompt already includes environment-provided valid actions, raw
action grounding keeps the SFT target aligned with the final executor
interface. The model should learn to copy one legal action exactly, rather than
learning a candidate-list position such as `A3` that has no stable meaning
across prompts.

The collector also gives the Sub agent a short recent execution history with
reward, score delta, and whether the observation changed. By default, the
candidate action set is the full environment-provided valid action list. The
collector does not filter `focus on ...`, graph actions, or repeated actions in
the benchmark-faithful path.

Sub execution is bounded by `--max-steps-per-subtask` so a stale contract cannot
consume the entire episode. The Sub agent can also hand control back explicitly:
`complete` for a satisfied contract, `blocked` when no valid action can make
progress, and `need_replan` when the contract no longer matches the state.

Optional ranking and truncation flags are kept only for debugging, ablations,
or token-budgeted pilots. The benchmark-faithful default is to pass the full
valid action list through unchanged.

Useful v2 options:

- `--history-limit 6`: number of recent Sub executions shown back to Kimi.
- `--max-steps-per-subtask 6`: maximum executor actions before returning to
  Main for replanning.
- `--rank-valid-actions`: rank valid actions by simple task-action heuristics.
- `--max-valid-actions N`: truncate the valid action list after optional
  ranking; default `0` means no truncation.

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

For default raw-action rollouts, Sub SFT targets are executable actions:

```text
[action]look around[/action][subtask_done]true|false[/subtask_done][handoff]continue|complete|blocked|need_replan[/handoff]
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
