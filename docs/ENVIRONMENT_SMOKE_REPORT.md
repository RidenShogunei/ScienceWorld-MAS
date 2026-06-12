# Environment Smoke Report

## Purpose

Verify that the hierarchical policy can switch between Main and Sub LoRA
adapters, step the official ScienceWorld environment, and save complete
episode trajectories.

## Configuration

- Base model: Qwen2.5-1.5B-Instruct
- Checkpoint: 128-sample SFT pilot
- Environment: ScienceWorld 1.2.3
- Split/task: dev / boil
- Episodes: 1
- Step limit: 8

## Result

| Metric | Result |
| --- | ---: |
| Main/Sub format errors | 0 |
| Environment score | 0 |
| Action valid rate | 0% |
| Steps completed | 8 |

Main produced the sensible subtask `Navigate to kitchen`, but Sub repeatedly
generated commands such as `use remote control`, which were not valid in the
current state.

## Interpretation

The software and adapter-switching path work. The 128-example pilot learned
the communication format but not the exact ScienceWorld action language. This
is expected from the offline pilot metrics and supports running a substantially
larger SFT baseline before drawing conclusions about RL.

