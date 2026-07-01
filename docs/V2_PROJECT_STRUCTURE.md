# V2 Project Structure

This branch is the cleaned-up v2 line. `main` preserves the historical
contract/MGRPO work; this branch intentionally does not carry the old
root-level scripts, generated data, checkpoints, or legacy tests.

## Target Layout

```text
src/scienceworld_mas/
  bench/          benchmark protocol, metrics, official reward semantics
  env/            thin ScienceWorld wrappers
  agents/         System1/System2 interfaces and parsers
  data/           expert transition builders and dataset manifests
  training/       SFT, BC, offline RL, online RL entry points
  evaluation/     fixed-list evaluation and task-level reports

configs/
  bench/          fixed protocol configs
  data/           dataset build configs
  training/       stage-specific training configs
  evaluation/     fixed eval configs

docs/
  BENCH_PROTOCOL.md
  V2_PROJECT_STRUCTURE.md
  experiments/    archived experiment reports
```

## Migration Rules

1. New code goes under `src/scienceworld_mas/`.
2. Do not add root-level runnable scripts.
3. Historical v1 code should be read from `main` or git history, not copied
   back into this branch.
4. Contract protocol, MrlX-like MGRPO, and joint Main+Sub RL remain ablations
   unless fixed-list evaluation beats the v2 baseline.
5. Large checkpoints, raw rollouts, provider caches, and temporary aggregate
   files stay out of git.

## First Milestone

The first milestone is a reproducible baseline runner with:

- fixed episode list loading
- official score preservation, including negative scores
- per-task evaluation report
- strict pass@1 single-rollout evaluation

The next milestone is:

- System1/System2 policy adapters
- System1 SFT + System2 BC baseline
- System2-only RL entry point
