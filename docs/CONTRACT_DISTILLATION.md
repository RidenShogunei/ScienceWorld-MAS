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
[action]...[/action][subtask_done]true|false[/subtask_done][handoff]continue|complete[/handoff]
```

## Kimi Code Usage

Kimi Code keys are intended for coding agents. The generator supports this path
through the official Kimi Code CLI and the temporary `KIMI_MODEL_*` model
configuration. Do not commit API keys.

```powershell
$env:KIMI_CODE_API_KEY = "..."
python generate_contract_sft_data.py `
  --provider kimicode-cli `
  --limit 100 `
  --cache-dir artifacts/contract_distill_cache `
  --output-dir data/contract_sft_kimicode_100
```

The CLI path is auto-discovered from `PATH` or
`%USERPROFILE%\.kimi-code\bin\kimi.exe`. If needed, pass it explicitly:

```powershell
python generate_contract_sft_data.py `
  --provider kimicode-cli `
  --kimicode-cli-path "$env:USERPROFILE\.kimi-code\bin\kimi.exe" `
  --limit 100
```

## Moonshot/OpenAI-Compatible Usage

If you have a regular Moonshot/Kimi OpenAI-compatible Chat Completions key
rather than a Kimi Code key, use the SDK provider. The generator defaults to
`https://api.moonshot.ai/v1` and reads the key from `MOONSHOT_API_KEY`.

```powershell
$env:MOONSHOT_API_KEY = "..."
python generate_contract_sft_data.py `
  --provider kimi `
  --model kimi-k2.6 `
  --limit 100 `
  --output-dir data/contract_sft_moonshot_100
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
trajectories, keeping supervision auditable. `handoff` is derived deterministically:
`complete` when the expert subtask is done, otherwise `continue`.

## Caching And Failures

Each distilled contract is cached by `(source_index, trajectory_step)` under
`artifacts/contract_distill_cache` by default. Re-running the command reuses
cached contracts and avoids duplicate API cost. Validation failures are written
to `artifacts/contract_distill_failures` for inspection.

## Quality Audit

Run the audit before using a generated batch for SFT:

```powershell
python audit_contract_sft.py `
  --input-dir data/contract_sft_kimicode_100 `
  --output artifacts/contract_sft_kimicode_100_audit.json
```

Important fields:

- `parse_ok_rate`: generated contracts are valid JSON/schema.
- `field_nonempty_rate`: required communication fields are populated.
- `exact_expert_action_prefix_rate`: `action_guidance` begins with official
  expert actions exactly. This should be `1.0`; the generator also enforces it
  during post-processing.
