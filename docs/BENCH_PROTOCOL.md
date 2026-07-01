# Bench Protocol

This branch resets the project around a bench-faithful ScienceWorld protocol.
The goal is to separate the reproducible benchmark baseline from exploratory
contract/MGRPO ablations.

## Primary Protocol

The main line follows the Multi-Square/System1-System2 framing:

- **System1/Main planner**: trained with supervised learning on expert
  high-level subtasks.
- **System2/Sub executor**: trained with behavior cloning, then offline/online
  RL using ScienceWorld environment feedback.
- **Joint Main+Sub RL**: treated as an ablation, not the default benchmark
  route.

## Metrics

The primary metric is official ScienceWorld mean score on a fixed episode list.

Secondary metrics:

- success rate
- per-task mean score
- action-valid rate
- format-error rate
- mean steps

Training rollout mean is a diagnostic only. It is not a replacement for fixed
episode evaluation.

## Reward Policy

RL should use official ScienceWorld score/reward as the main signal. Negative
scores must be preserved because terminal cliff failures carry information that
is lost when scores are clamped to zero.

Format validity and action validity are still important, but in the v2 baseline
they should be logged as metrics or used as explicit filters. They should not
silently redefine the benchmark reward unless an experiment is clearly marked
as reward-shaping ablation.

## Episode Policy

Evaluation uses:

```text
artifacts/eval/dev_stratified_k5_seed123.json
```

Training rollout sampling should be task-balanced or curriculum-controlled.
Randomly sampling a small subset of tasks per iteration is not treated as a
stable benchmark protocol.
