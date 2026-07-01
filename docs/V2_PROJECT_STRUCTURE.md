# V2 Project Structure

This branch is the cleaned-up v2 line. `main` preserves the historical
contract/MGRPO work; this branch introduces a new project shape before moving
large pieces of code.

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
2. Old root-level scripts stay available until a v2 replacement exists.
3. Once a v2 replacement is tested, the old script should move to `legacy/` or
   be removed from this branch.
4. Contract protocol, MrlX-like MGRPO, and joint Main+Sub RL remain ablations
   unless fixed-list evaluation beats the v2 baseline.
5. Large checkpoints, raw rollouts, provider caches, and temporary aggregate
   files stay out of git.

## First Milestone

The first milestone is not a new model score. It is a reproducible baseline
runner with:

- fixed episode list loading
- official score preservation, including negative scores
- per-task evaluation report
- System1 SFT + System2 BC baseline
- System2-only RL entry point
