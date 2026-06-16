# Contract Communication Distillation

## Motivation

The original Main-to-Sub interface is only a one-line subtask:

```text
[subtask]Navigate to kitchen[/subtask]
```

ScienceWorld execution requires grounded information that is absent from that
message: target objects, success conditions, useful action styles, location
hints, and fallback behavior. Contract distillation expands expert
Multi-Square steps into structured communication while preserving official
expert action labels.

## Contract Format

```json
{
  "goal": "...",
  "subgoal": "...",
  "rationale": "...",
  "target_objects": ["..."],
  "location_hint": "...",
  "required_tools": ["..."],
  "success_condition": "...",
  "action_guidance": ["..."],
  "fallback_if_blocked": "..."
}
```

The Main model learns to emit:

```text
[contract]{...}[/contract]
```

The Sub model receives the contract plus current observation and still learns
the official expert action:

```text
[action]...[/action][subtask_done]true|false[/subtask_done]
```

## Kimi/Moonshot Usage

The Kimi API is OpenAI Chat Completions compatible. The generator defaults to
the documented base URL `https://api.moonshot.ai/v1` and reads the key from
`MOONSHOT_API_KEY`. Do not commit API keys.

```powershell
$env:MOONSHOT_API_KEY = "..."
python generate_contract_sft_data.py `
  --provider kimi `
  --model kimi-k2.6 `
  --limit 100 `
  --cache-dir artifacts/contract_distill_cache `
  --output-dir data/contract_sft_kimi_100
```

For code-path testing without API calls:

```powershell
python generate_contract_sft_data.py `
  --provider mock `
  --limit 10 `
  --output-dir data/contract_sft_mock10
```

## Design Constraint

Kimi distills communication contracts only. It does not replace environment
action labels. Actions and `subtask_done` remain from Multi-Square expert
trajectories, keeping supervision auditable.

## Caching And Failures

Each distilled contract is cached by `(source_index, trajectory_step)` under
`artifacts/contract_distill_cache` by default. Re-running the command reuses
cached contracts and avoids duplicate API cost. Validation failures are written
to `artifacts/contract_distill_failures` for inspection.
