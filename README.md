# ScienceWorld-MAS v2

This branch is a clean, bench-faithful rewrite of the ScienceWorld-MAS project.
The historical contract/MGRPO code remains available on `main` and in git
history; it is intentionally not carried in this branch.

## Goal

Build a reproducible ScienceWorld hierarchical-agent baseline that follows the
Multi-Square/System1-System2 framing:

```text
System1 planner  -> high-level subgoal
System2 executor -> executable ScienceWorld action
```

The v2 baseline keeps these boundaries explicit:

- System1/Main planner: supervised learning on expert high-level subtasks.
- System2/Sub executor: behavior cloning, then offline/online RL.
- Joint Main+Sub RL: ablation only, not the default benchmark route.

## Active Layout

```text
src/scienceworld_mas/       v2 package code
configs/bench/             canonical benchmark protocol config
docs/BENCH_PROTOCOL.md     evaluation, reward, and training rules
docs/V2_PROJECT_STRUCTURE.md
tests/                     v2 tests only
```

The canonical v2 config is:

```text
configs/bench/bench_faithful_v2.json
```

## Protocol

Read [docs/BENCH_PROTOCOL.md](docs/BENCH_PROTOCOL.md) first. The short version:

- Primary metric: official ScienceWorld mean score on a fixed episode list.
- Secondary metrics: success rate, per-task score, action-valid rate,
  format-error rate, and mean steps.
- RL reward source: official ScienceWorld score/reward.
- Negative scores are preserved; they are not clamped to zero.
- Rollout mean during training is diagnostic only.

## Development

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the current v2 tests:

```powershell
python -m pytest tests -q
```

## Migration Status

This branch currently contains the v2 protocol skeleton and reward semantics.
The next migrations should add, in order:

1. Thin ScienceWorld environment wrapper under `src/scienceworld_mas/env/`.
2. Fixed-list evaluator with per-task reporting under `src/scienceworld_mas/evaluation/`.
3. System1/System2 data builders under `src/scienceworld_mas/data/`.
4. System1 SFT and System2 BC/RL training entry points under
   `src/scienceworld_mas/training/`.

Avoid adding new root-level scripts. New runnable entry points should live under
`src/scienceworld_mas/` and be exposed through small, named CLI modules.

## Upstream

- Multi-Square project: https://park-sangeun.github.io/Multi-Square/
- Multi-Square dataset: https://huggingface.co/datasets/sangeun-park/Multi-Square
- ScienceWorld: https://github.com/allenai/ScienceWorld
