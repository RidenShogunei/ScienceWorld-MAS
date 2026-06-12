# ScienceWorld SFT Pilot Report

## Setup

- Base model: Qwen2.5-1.5B-Instruct
- Training: 4-bit LoRA
- Samples: 128 Main and 128 Sub
- Validation samples: 32 per agent
- Updates: 32 per agent
- Evaluation: first 20 held-out examples per agent

This pilot validates the training and evaluation pipeline. It is not intended
as a competitive benchmark result.

## Results

| Agent | Format valid | Exact match | Additional metric |
| --- | ---: | ---: | ---: |
| Main | 100% | 20% | N/A |
| Sub | 100% | 0% | action exact 15%, done accuracy 40% |

Validation loss was 0.889 for Main and 0.371 for Sub.

## Interpretation

The small pilot learned the required output schemas and some task content.
Main already recovered several exact expert subtasks. Sub frequently generated
semantically related actions that are not exact ScienceWorld commands, such as
using an object instead of issuing the required `move` action. It also
over-predicted `subtask_done=false`.

This separates three capabilities that should remain distinct in later
reports:

1. Output format validity.
2. Exact environment action and termination prediction.
3. Full executable episode success.

The full SFT run is justified because the current bottleneck is environment
action language and subtask-boundary behavior, both directly supervised by the
provided Multi-Square expert dataset.

