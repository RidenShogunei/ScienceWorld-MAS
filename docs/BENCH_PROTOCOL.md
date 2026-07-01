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

## Comparable Metrics

The comparable ScienceWorld metric is official mean score on a fixed episode
list. This is the metric intended for comparison with Multi-Square/ScienceWorld
results.

Do not compare models using success rate, best-of-N success, sampled rollout
mean, or reward-shaped scores.

## Diagnostic Metrics

These are useful for debugging but are not the headline comparable score:

- success rate
- per-task mean score
- action-valid rate
- format-error rate
- negative-score rate
- mean steps

Training rollout mean is a diagnostic only. It is not a replacement for fixed
episode evaluation.

## Rollout Policy

Comparable evaluation uses strict pass@1:

- one attempt per episode
- no retry on failure
- no best-of-N or majority vote
- fixed episode list
- official final ScienceWorld score is recorded for that single rollout

Sampling-based, best-of-N, curriculum, or training-rollout evaluations must be
reported as ablations or diagnostics, not as directly comparable benchmark
results.

## Reward Policy

RL should use official ScienceWorld score/reward as the main signal. Negative
scores must be preserved because terminal cliff failures carry information that
is lost when scores are clamped to zero.

Format validity and action validity are still important, but in the v2 baseline
they should be logged as metrics or used as explicit filters. They should not
silently redefine the benchmark reward unless an experiment is clearly marked
as reward-shaping ablation.

## Scoring Contract

All evaluation code should aggregate results through
`scienceworld_mas.bench.compute_benchmark_score`.

The scorer intentionally:

- averages official final episode scores directly
- preserves negative scores in the mean
- reports success as `score >= 100`
- reports task-level means over the same episode records
- treats action validity and format errors as secondary health metrics

Strict pass@1 environment rollout is implemented in
`scienceworld_mas.evaluation.run_episode` and
`scienceworld_mas.evaluation.evaluate_episodes`.

The command-line smoke entry point is:

```text
python -m scienceworld_mas.evaluation.cli --policy gold ...
```

`gold` is an oracle environment/evaluator smoke policy. It must not be reported
as a model result.

## Episode Policy

Evaluation uses:

```text
artifacts/eval/dev_stratified_k5_seed123.json
```

Training rollout sampling should be task-balanced or curriculum-controlled.
Randomly sampling a small subset of tasks per iteration is not treated as a
stable benchmark protocol.

## Data Policy

Multi-Square ScienceWorld data is represented as role-specific transitions:

- System1 transition: `task_description + observation -> target_subgoal`
- System2 transition: `subgoal + observation -> target_action + subgoal_done`

Environment `reward`, cumulative `score`, and `done` metadata are retained in
the transition records. Splits are deterministic and grouped by normalized task
family for System1 and normalized subgoal for System2 to avoid step-level
leakage.

## Supervised Warm Start

The supervised baseline is intentionally role-separated:

- System1 SFT trains only the high-level planner target.
- System2 behavior cloning trains only executable actions and
  `subgoal_done`.

Both use the same processed transition root:

```text
data/processed/v2/{system1,system2}/{train,val,test}.jsonl
```

The package entry point is:

```text
python -m scienceworld_mas.training.supervised --role system1 ...
python -m scienceworld_mas.training.supervised --role system2 ...
```

Supervised training writes `best/` and `last/` LoRA adapter directories. Fixed
episode evaluation should load `best/` unless an experiment explicitly reports
that it is evaluating `last/`.

## Model Evaluation Policy

The default model policy is hierarchical:

1. System1 receives `task_description + observation` and emits one subgoal.
2. System2 receives `subgoal + observation` and emits
   `[action]...[/action][subgoal_done]true|false[/subgoal_done]`.
3. System1 is called again only after System2 marks the current subgoal done.

The evaluation runner still passes valid actions and history to every policy
for diagnostics and future ablations, but the default v2 supervised model
policy does not include them in the prompt because they are not present in the
Multi-Square transition training distribution.

The command-line entry point is:

```text
python -m scienceworld_mas.evaluation.cli --policy hf-hierarchical ...
```
