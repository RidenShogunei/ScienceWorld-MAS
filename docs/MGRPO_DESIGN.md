# ScienceWorld M-GRPO Design

## Training Unit

For each identical ScienceWorld task and variation, sample `G` complete system
rollouts. One system rollout contains:

```text
one complete Main conversation
  -> zero or more Main decisions
  -> zero or more Sub invocations
       -> one or more Sub action generations
```

Main decisions are retained for local diagnostics and reward components, but
the Main policy gradient is computed over the complete Main conversation.
A Sub invocation is the alignment unit used by M-GRPO. Individual action-token
lengths are handled separately with ordinary token padding and masks.

## Group Advantage

Complete rollouts from the same task and variation form one GRPO group:

```text
A_system = (R_system - mean(R_group)) / (std(R_group) + epsilon)
```

Rollouts from different tasks or variations must never share the same
normalization group.

## Sub-Trajectory Alignment

Every system rollout is mapped to exactly `M` Sub-invocation slots:

- `N = 0`: create `M` empty slots with `loss_mask=0`.
- `0 < N < M`: retain all invocations and sample existing indices with
  replacement for the remaining slots.
- `N >= M`: sample `M` invocation indices without replacement.

Alignment duplicates training indices, not environment trajectories. Original
observations, actions, rewards, and scores remain immutable.

After computing the composite Sub reward, group-relative normalization is
performed separately over the valid aligned Sub slots for the same query.
Main advantages are never directly reused as Sub advantages.

## Policy Objective

Every generation stores its prompt messages, completion token IDs, and
old-policy token log-probabilities. The update recomputes current log-probs and
uses the clipped token-level PPO/GRPO surrogate. Padding tokens and empty
aligned Sub slots are masked. A frozen SFT reference can add a sampled-token KL
penalty controlled by `beta`.

## Rewards

Main reward is attached to the complete Main trajectory and includes:

- normalized final environment score;
- positive score progress made by delegated subtasks;
- output-format validity;
- penalties for repeated or no-progress subtasks.

Each Sub invocation receives:

- replicated global environment outcome;
- local score progress;
- exact environment-action validity;
- format validity;
- penalties for repeated actions and observably premature `subtask_done=true`.

Natural-language subtask completion is not directly observable in
ScienceWorld. The implementation therefore does not invent a completion
oracle. It uses only environment score, environment termination, action
validity, and generated completion declarations.

## Planned Ablations

1. SFT baseline.
2. Main-only GRPO.
3. Sub-only GRPO.
4. Joint GRPO without invocation alignment.
5. Joint M-GRPO with invocation alignment.
6. Joint M-GRPO without local Sub reward.

All variants must use the same SFT initialization, task groups, rollout budget,
and held-out dev checkpoint selection.
