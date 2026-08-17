# 0009 - Batch snapshot inference and reset long-lived ROCm processes

**Status:** Proposed on `feat/stacked-snapshot-inference` (2026-08-17)

## Context

The vector rules engine made self-play rollout policy-bound. A rollout can use
several historical PPO snapshots, and the original implementation evaluated
each snapshot independently at every decision boundary. Every evaluation
computed actor and critic outputs, copied values and log-probabilities that an
opponent never trains on, and synchronised the whole device before returning.

The persistent snapshot pool also outlives individual rollouts. Long-lived
Windows ROCm processes showed a second, independent problem: snapshot calls
that initially took roughly 2-3 ms drifted toward 4 ms after 25 updates even
though explicit garbage collection and cache emptying ran between updates.

## Decision

Snapshot opponents use an actor-only policy path. They produce only sampled
actions; critic values and action log-probabilities remain exclusive to the
learning policy.

When a rollout activates several architecture-compatible snapshots, their
parameters are stacked with `torch.func.stack_module_state` and evaluated with
`vmap`. Ragged row groups are padded with observations whose only legal action
is `END_TURN`, so padding cannot affect real game rows. One host-to-device
copy, actor evaluation, sample, device-to-host copy, and synchronization serve
the complete group.

A one-snapshot rollout uses a more efficient joint path. The learning model
keeps its full actor/critic evaluation, the snapshot keeps its actor-only
evaluation, their unequal batches execute consecutively on one stream, and
both device-to-host copies complete behind one synchronization. This avoids
the padding and duplicated critic work that made a two-model `vmap` slower.

Acting-player observations are materialized only for rows currently controlled
by snapshots. The normal learning-policy observation remains a full resident
population because every live game may contribute training rows over time.

Training arcs run in fresh Python processes of at most 25 updates. The chunk
orchestrator:

- resumes the latest checkpoint, including optimizer, scheduler, and snapshot
  manifest;
- restores Python, NumPy, Torch CPU, and accelerator RNG streams at the exact
  post-update boundary;
- pins every process to one final learning-rate and self-play horizon;
- writes an atomic durable manifest before and after each chunk;
- preserves each chunk's log and metrics;
- uses cheap evaluation between chunks and the requested evaluation on the
  final chunk;
- aborts unless the process exits successfully and creates exactly the
  expected checkpoint.

The persistent PFSP pool remains unchanged. Limiting active snapshots changes
only how many members are sampled for one resident population; every pool
member remains eligible on later updates.

## Rejected alternatives

**CPU snapshot inference.** Direct measurements were 50-169 ms per call for
the relevant batch sizes, versus 2-4 ms on the GPU.

**Full-policy stacked fusion.** Stacking the learner and one snapshot forced
both models through the larger padded batch and evaluated a useless snapshot
critic. It reached a 3.9 s warm rollout, compared with 3.4 s for the joint
unequal-batch path.

**Allocator cleanup without process resets.** Garbage collection,
`empty_cache()`, and synchronization already ran every update. They did not
prevent the measured latency drift.

## Verification

The implementation requires:

- numerical equality between actor-only logits and the actor output of the
  full policy;
- stacked logits equal to independent model logits;
- joint current-policy values equal to a normal full forward pass;
- legal sampled actions for ragged and joint batches;
- selected-row observations equal to slicing full-population observations;
- complete vector self-play rollouts and resumable chunk manifests;
- full repository tests and mandatory review;
- a cap-1/2/3 strength ablation before changing the shipped active-snapshot
  default.

Fresh-process measurements from the same update-213 checkpoint reduced warm
rollout from 7.2 s with independent cap-3 inference to 5.2 s with stacked
cap-3 inference, 4.2 s with actor-only cap-1 inference, and 3.4 s with the
joint cap-1 path. The last configuration spends about 6.5 s on rollout plus
PPO, meeting the optimization target before strength selection.

## Consequences

- Snapshot inference scales with active model groups rather than one
  synchronization per model.
- Cap 1 has a dedicated fast path without weakening the persistent PFSP pool.
- Cap 2 and cap 3 retain broader within-update opponent diversity through the
  stacked path.
- Process startup is paid once per chunk rather than once per update, while
  long-process ROCm drift is bounded.
- The active-snapshot cap remains an empirical strength/speed choice; this ADR
  does not change its default before the locked ablation.
