# 0008 - Group snapshot self-play inference in vector rollout

**Status:** Accepted on `feat/vector-self-play-dense` (2026-08-16)

## Context

ADR 0006 moved all fixed opponents into the Numba vector runner, but snapshot
self-play still forced the object/worker architecture. Training already owns a
bounded pool of immutable PPO snapshots, a self-play ratio schedule, PFSP
weights, EMA result tracking, and a resumable on-disk manifest.

The missing piece was vector rollout inference for games assigned to those
snapshot models.

## Decision

Extend the per-game opponent assignment:

- codes 0-2 remain Random, Heuristic, and Simple;
- codes 3+ identify a snapshot in the current immutable pool.

At the beginning of a resident population, each game is assigned either a
fixed opponent or snapshot according to the current self-play ratio. Snapshot
selection uses the PFSP weights frozen for that update. Assignments remain
stable for the complete game.

The persistent pool remains log-spaced and PFSP-weighted, but one vector
population activates at most three snapshot models by default. The active
subset is sampled without replacement using PFSP weights, then self-play games
are distributed across that subset. A different subset is selected each
update, so the full pool remains eligible while inference fragmentation is
bounded. `--vec-max-active-snapshots 0` restores unlimited activation.

Only snapshots assigned at least one resident game are reconstructed. Their
evaluators are cached for the runner lifetime and kept on the simulation
device. Opponent decision rows are grouped by snapshot id and sent through one
batched `VectorPolicyEvaluator` call per active snapshot. Fixed rows continue
through the Numba selector.

Snapshot policies receive a separate reusable observation buffer oriented to
the acting player. The learning policy's rollout buffer remains oriented to
`training_player`; reusing it for snapshot decisions would invert private and
opponent zones whenever the snapshot is acting.

Result accounting uses the existing snapshot names, so `OpponentPool` EMA,
PFSP summaries, geometric eviction, checkpoint manifests, and resume behavior
remain unchanged.

The existing dense vector rollout stores only the current policy's decisions.
Snapshot actions are behavior of the environment opponent and therefore do not
enter PPO minibatches.

## Verification

- assignment tests cover fixed fallback, self-play ratio, and PFSP weighting;
- active-subset tests enforce the per-rollout cap and full-pool coverage across
  updates;
- trainer plumbing passes live and restored snapshot pools to the vector
  runner;
- all-snapshot vector games complete with legal actions and per-snapshot
  results;
- multi-update detached training proves snapshot creation, later-use, PFSP
  updates, checkpoint manifests, and resume;
- full repository and mandatory review gates are required before merge.

## Consequences

- Fixed, scheduled self-play, and PFSP curricula can use the same vector
  simulation path.
- Policy inference cost grows with the number of distinct snapshots active in
  one iteration, not directly with the number of self-play games.
- The snapshot cap bounds resident model memory.
- The active-model cap bounds policy-call fragmentation independently from the
  persistent snapshot-pool cap.
- Building snapshot models once per update adds setup work but avoids
  per-game model construction.
