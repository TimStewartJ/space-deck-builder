# 0009 - Batch snapshot inference and reset long-lived ROCm processes

**Status:** Accepted on `feat/stacked-snapshot-inference` (2026-08-17)

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
- pins every process to explicit learning-rate and self-play horizons, which
  may extend beyond a shorter experiment stop update;
- writes an atomic durable manifest before and after each chunk;
- preserves each chunk's log and metrics;
- uses a single-process cheap evaluation between chunks and the requested
  worker topology on the final evaluation;
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

## Active-snapshot selection

The architecture-selection cap pilot resumed the same update-213 checkpoint
for 50 updates at 2,000 episodes/update and continuation seed 0. Every arm
retained the original 400-update schedules, used fresh processes after at most
25 updates, and finished with 5,000 games against each fixed opponent. This is
a paired engineering pilot, not the three-independent-seed result required for
a reported model by `eval_protocol.md`.

| Active cap | Warm rollout | Warm rollout + PPO | Heuristic | Overall | Elo |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.27 s | 6.28 s | 82.92% | 93.17% | 1092 |
| 2 | 5.48 s | 8.72 s | 84.14% | 93.51% | 1098 |
| 3 | 5.42 s | 8.47 s | 85.20% | 94.01% | 1093 |

Elo used 1,000 games/pair and anchored the fixed-opponent update-400 baseline
at 1000. The cap-1, cap-2, and cap-3 checkpoints were statistically close in
cross-play: cap 1 scored 51.3% against cap 2 and 48.8% against cap 3; cap 2
scored 51.8% against cap 3.

Cap 1 met the aggressive latency target, but its seed-0 Heuristic point
estimate was 2.28 percentage points below cap 3, beyond the pilot's one-point
guardrail. Cap 2 had no speed advantage over cap 3. Therefore this pilot does
not justify changing the shipped cap-3 default; cap 1 remains an explicit
fast-throughput option. A future reported strength claim still requires three
independent training seeds.

Against the degraded long-lived cap-3 process, the selected cap-3 design
reduced warm rollout from 12.41 s to 5.42 s and rollout-plus-PPO from 16.23 s
to 8.47 s. This is a 2.29x rollout speedup and 1.92x end-to-end update speedup.

## Consequences

- Snapshot inference scales with active model groups rather than one
  synchronization per model.
- Cap 1 has a dedicated fast path without weakening the persistent PFSP pool.
- Cap 2 and cap 3 retain broader within-update opponent diversity through the
  stacked path.
- Process startup is paid once per chunk rather than once per update, while
  long-process ROCm drift is bounded.
- Cap 3 remains the default because it preserved the strongest fixed-opponent
  result at the same throughput as cap 2.

## Three-seed follow-up

The protocol-compliant 400×2,000 follow-up retained cap 3 and 25-update process
chunks. Across seeds 0, 1, and 2 it measured 5.22s warm rollout and 8.18s total
warm update time. Fixed-opponent performance was statistically tied with the
fixed-only control, while all three self-play seeds ranked above all three
control seeds in cross-play Elo. The result and preserved artifacts live in
[`results/optimized_selfplay_baseline.md`](../../results/optimized_selfplay_baseline.md).
