# Snapshot inference optimization

**Branch:** `feat/stacked-snapshot-inference`  
**Implementation commits:** `1189f28`, `6674b0c`, `2a9c846`  
**Experiment root:** `results/snapshot_optimization/cap_ablation_20260817_010705`

> **Scope:** This is a paired architecture-selection pilot from one shared
> update-213 checkpoint with continuation seed 0. It uses the protocol-sized
> gauntlet and Elo sample counts, but it is not a three-independent-training-
> seed result and must not be reported as one.

The three-independent-seed follow-up is now complete; see
[`optimized_selfplay_baseline.md`](optimized_selfplay_baseline.md).

## Result

The production choice is:

- actor-only historical snapshot policies;
- stacked, padded inference for up to three active snapshots;
- a joint one-snapshot fast path with one synchronization;
- three active snapshots by default;
- a persistent ten-snapshot PFSP pool;
- a fresh Python process every 25 updates;
- checkpointed optimizer, scheduler, snapshot manifest, and all RNG streams.

This preserves the stronger cap-3 curriculum while making the degraded
long-run workload roughly twice as fast.

## Performance

All cap arms resumed update 213, ran 50 updates at 2,000 episodes/update with
seed 0, retained the original update-400 LR/curriculum horizons, and excluded
the first cold update of each fresh process from warm means.

| Active snapshots | Rollout | PPO | Total/update | Episodes/s |
|---:|---:|---:|---:|---:|
| 1 | 3.27 s | 3.01 s | 6.28 s | 613 |
| 2 | 5.48 s | 3.24 s | 8.72 s | 367 |
| 3 | 5.42 s | 3.05 s | 8.47 s | 370 |
| Old cap-3 process, updates 2-25 | 7.09 s | 3.08 s | 10.17 s | 282 |
| Old cap-3 process, updates 26-213 | 12.41 s | 3.82 s | 16.23 s | 161 |

Selected cap 3 versus the degraded process:

- rollout: **12.41 s to 5.42 s** (56% lower, 2.29x throughput);
- complete update: **16.23 s to 8.47 s** (48% lower, 1.92x faster);
- snapshot opponent work: **7.29 s to 2.67 s**;
- long-process ROCm drift is bounded by 25-update process chunks.

Cap 1 remains available when throughput matters more than the locked
fixed-opponent threshold. It reached 3.27 s rollout and 6.28 s total/update.

## Protocol-sized selection gauntlet

Each checkpoint played 5,000 games against each fixed opponent.

| Active snapshots | Random | Heuristic | Simple | Overall |
|---:|---:|---:|---:|---:|
| 1 | 99.78% | 82.92% | 96.80% | 93.17% |
| 2 | 99.80% | 84.14% | 96.58% | 93.51% |
| 3 | **99.82%** | **85.20%** | **97.02%** | **94.01%** |

Cap 1's seed-0 point estimate was 2.28 percentage points lower against
Heuristic than cap 3, beyond the pilot's one-point guardrail. Cap 2 offered no
speed improvement over cap 3. This pilot therefore provides no evidence to
change the existing cap-3 default. A reportable strength conclusion still
requires three independent training seeds and mean confidence intervals.

The PFSP curriculum stayed active across the sweep. Caps 1, 2, and 3 faced
35, 42, and 47 distinct historical snapshots respectively. Zero-weight and
geometrically evicted snapshots are not expected to appear in every finite
run; cap 3 produced the broadest observed coverage.

## Cross-play Elo

The tournament used 1,000 games/pair. The fixed-opponent update-400 baseline
is anchored at Elo 1000.

| Rank | Checkpoint | Elo |
|---:|---|---:|
| 1 | Cap 2, update 263 | 1098 |
| 2 | Cap 3, update 263 | 1093 |
| 3 | Cap 1, update 263 | 1092 |
| 4 | Self-play start, update 213 | 1073 |
| 5 | Fixed-opponent update-400 baseline | 1000 |

The cap checkpoints are effectively tied in head-to-head play:

- cap 1 vs cap 2: 51.3%;
- cap 1 vs cap 3: 48.8%;
- cap 2 vs cap 3: 51.8%.

All three continuation checkpoints beat the update-213 start and the fixed
baseline in this tournament. Their mutual Elo differences are small, so the
conservative engineering choice is to retain cap 3 rather than trade observed
Heuristic performance for cap-1 speed on only one continuation seed.

## Validation

- 544 repository tests passed.
- A detached two-chunk smoke restored Python, NumPy, Torch CPU, and ROCm RNG
  state and preserved both schedule horizons.
- The mandatory code-review gate reported no remaining significant issues.
- Raw logs, metrics, checkpoints, manifests, and both Elo runs are retained
  under the experiment root.
