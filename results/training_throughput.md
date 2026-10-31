# Training throughput: graph-replayed rollouts and device-side PPO bookkeeping

**Status:** Complete (2026-10-30). Design and rejected alternatives:
[`docs/decisions/0017-graphed-rollout-inference.md`](../docs/decisions/0017-graphed-rollout-inference.md).
**Branch:** `perf/training-throughput`.
**Evidence:** [`results/training_throughput/`](training_throughput/), indexed
in its `README.md`. The run folders themselves, including every checkpoint
and the timing fixture, are archived on the project machine at
`D:\space-deck-builder-archive\training-throughput-20261030`; a path below
under `D:\sdb-bench` is under that folder's `sdb-bench`.

## Result

A warm reference-recipe update takes **2.17 s instead of 9.64 s** on one
training lane (4.4x), and an interim evaluation of 3,200 games takes about
**1.2 s instead of about 23 s**. A complete 400-update reference run from
scratch took **19.3 minutes**, including every 25-update process restart and
the locked final gauntlet, with a second training process sharing the
machine for its first eight minutes; the published runs of the same recipe
took about an hour. The retrained seed 0 learns the same curves as the
published seed 0 and lands inside the published seeds' strength range.

Three lanes side by side reach **about one update per second** in total, the
GPU's limit on this machine. A process no longer slows with age, so the
presets now run in 100-update processes instead of 25-update ones, which
saves about 10% of an arc's time.

## Method

Timing runs resumed the same checkpoint, update 200 of the phase 1
`continue-selfplay-seed0` seed-0 run (`D:\space-deck-builder-archive\strength-yardstick-20261027\runs\continue-selfplay-seed0_seed0\chunks\models`),
with its ten-snapshot pool, and trained 12 updates through Task Scheduler,
pinned to eight logical performance-core CPUs (0-7) with nothing else on
them. Means cover updates 202-211, excluding the first update of the process.
Platform: Windows, RX 9070, torch 2.11.0+rocm7.13.0a20260408.

The branch column was measured at `cd9d034`. The final code, which keeps PPO
minibatches eager and fixes a memory leak, measured 2.13-2.28 s per update in
the same setup; the spread is the PPO update's variation between processes
(see "Options measured and not taken").

## Performance

| | `master` (`2fb71e1`) | This branch | Speed-up |
|---|---:|---:|---:|
| Wall time per update | 9.64 s | 2.17 s | 4.4x |
| Rollout | 6.26 s | 1.23 s | 5.1x |
| PPO update | 3.18 s | 0.87 s | 3.7x |
| Evaluation, 3,200 games | about 23 s (object engine, 16 workers) | 1.1-1.7 s (vector engine) | about 15x |
| Final gauntlet, 15,000 games | 33-77 s (object engine, 16 workers, phase 1 runs) | 4.1 s | 8-19x |

Where a warm rollout's 1.2 s goes (one `[VectorRunner] graphed` line):
staging masks and observations 0.28 s, issuing copies and graph replays
0.37 s, fixed opponents 0.10 s (overlapped with the GPU), waiting for the GPU
0.27 s, engine transitions 0.10 s, over about 700 decision steps.

The PPO update is bound by GPU arithmetic: about 4.8 ms per 8,192-row
minibatch, most of it the network's backward pass. Capturing minibatches as
graphs did not change it (0.72 s against 0.68 s per update).

Measured host costs on this platform, which shaped the design:

| Operation | Host time |
|---|---:|
| Eager policy forward and sample, a few hundred rows | 2.5-3 ms |
| Graph replay, fixed part | about 40 us |
| Graph node, elementwise | about 3 us |
| Graph node, GEMM | about 38 us |
| Asynchronous host-to-device or device-to-host copy | 2-10 us |
| Device-to-device copy outside a graph | about 30 us |

## Validation: a complete reference run

The reference recipe (`--preset reference-selfplay`, seed 0, 400 updates in
25-update processes) was retrained from scratch on this branch at `c96a9dc`
with `train_reference.ps1` and production placement (logical CPUs 0-15). Run
root: `D:\sdb-bench\validate_seed0`.

Per-window means of the training metrics against the published seed 0
(`optimized_400x2000_20260817_230304\seed_0`, published seed first):

| Updates | Entropy | KL | Clip fraction | Explained variance | Win rate vs Heuristic |
|---|---|---|---|---|---|
| 26-50 | 0.774 / 0.745 | 0.0028 / 0.0029 | 0.024 / 0.027 | 0.846 / 0.852 | 74.0% / 75.5% |
| 51-100 | 0.751 / 0.723 | 0.0028 / 0.0029 | 0.025 / 0.026 | 0.850 / 0.852 | 81.0% / 81.5% |
| 101-200 | 0.729 / 0.725 | 0.0024 / 0.0025 | 0.022 / 0.022 | 0.853 / 0.852 | 84.8% / 84.5% |
| 201-300 | 0.718 / 0.715 | 0.0016 / 0.0017 | 0.013 / 0.013 | 0.863 / 0.862 | 87.1% / 86.9% |
| 301-400 | 0.723 / 0.715 | 0.0006 / 0.0007 | 0.003 / 0.003 | 0.867 / 0.866 | 88.4% / 88.3% |

Decisions per episode (145-146) and the rollout win rate (72-73%) also agree
in every window. The published run's win rates against Heuristic came from
object-engine evaluations and this run's from the vector engine, so the last
column also cross-checks the two evaluators.

Strength of the final checkpoint against the published self-play seeds:

| Measure | This run | Published seeds 0 / 1 / 2 |
|---|---:|---:|
| Final gauntlet, overall (15,000 games) | 95.6% | mean 94.4% ± 0.7 pp |
| Final gauntlet, Heuristic (5,000 games) | 87.9% | mean 86.4% ± 2.0 pp |
| Held-out league, pooled (24,000 games) | 77.6% | 79.2% / 73.9% / 78.5% |
| Elo at temperature 0, 4,000 games per pair | 1000 | 1011 / 977 / 1007 |

Head to head at temperature 0 the run won 48.1% of games against seed 0,
53.4% against seed 1 and 49.4% against seed 2; the published seeds differ
from one another by as much (seed 2 beats seed 1 55.4% of the time). The
league also replayed published seed 0 through the folded network and
reproduced its published counts exactly (3,084 / 3,321 / 3,111 / 3,055 /
3,092 / 3,345 wins of 4,000), so folding the input layer changed none of
its greedy decisions in 24,000 games.

## Running several lanes

Lanes are independent training processes sharing the GPU. Each resumed the
same checkpoint with the reference recipe and production placement; means
cover updates 205-226.

| Lanes | Seconds per update, per lane | Rollout | PPO update | Updates per second, all lanes |
|---:|---:|---:|---:|---:|
| 1 | 1.97 | 1.15 | 0.75 | 0.51 |
| 2 | 2.41 | 1.39 | 0.94 | 0.83 |
| 3 | 3.07 | 1.83 | 1.14 | 0.98 |
| 4 | 4.15 | 2.68 | 1.37 | 0.97 |
| 5 | 12.72 | 11.53 | 1.01 | 0.39 |

Three lanes reach the GPU's limit: a fourth adds nothing. Five collapse
because each lane runs four engine threads (`vec_threads`) plus its own main
thread, 25 busy threads on 16 logical CPUs; a parallel engine step then waits
for whichever thread was descheduled. CPU work from anything else slows the
lanes the same way, so tests, the league and audits should not run beside
them.

## GPU memory

A lane holds about 1.9 GB of dedicated GPU memory (Windows' per-process
counter): about 1.0 GB allocated, 0.8 GB of it the learner's rollout storage,
plus the caching allocator's reserve, the HIP context and the BLAS workspace.
A 15,000-game evaluation adds about 0.6 GB while it runs.

An earlier version leaked about 2.2 GB per lane. hipBLASLt keeps one
workspace of about 75 MB per stream and never frees it, and each graph
capture warmed up on a new side stream. Warm-ups now share one stream per
device.

## Process age and chunk length

Decision 0009 split training into fresh 25-update processes because eager
snapshot calls slowed as a process aged (from about 2-3 ms toward 4 ms within
25 updates). Graph replays do not. Three lanes each trained 400 updates in a
single process from the same checkpoint (updates 201-600):

| Updates | Lane 0 | Lane 1 | Lane 2 |
|---|---:|---:|---:|
| 201-300 | 2.50 s | 2.57 s | 2.75 s |
| 301-400 | 2.75 s | 2.55 s | 2.91 s |
| 401-500 | 2.51 s | 2.40 s | 2.58 s |
| 501-600 | 2.54 s | 2.59 s | 2.05 s |

The table gives each window's tenth-percentile seconds per update: another
test run used five CPU cores during parts of updates 301-450, which raised
medians (2.9-4.1 s) but not the fastest updates. Each lane's dedicated GPU
memory stayed at 1,899 MB from update 340 to update 466.

A process restart costs about 10 s: about 2 s between processes, 2 s of
startup and about 6 s of graph capture and kernel loading in its first
update. Three lanes then trained 100 updates of the reference recipe from
scratch through `scripts/run_queue.ps1`, once in 25-update processes and once
in a single process:

| Chunk length | Lane times | All three lanes |
|---|---:|---:|
| 25 updates (four processes per lane) | 331 / 351 / 349 s | 6.28 min |
| 100 updates (one process per lane) | 300 / 321 / 313 s | 5.61 min |

Three extra restarts per lane cost about 32 s, so the other lanes did not
absorb the restarts. The single-process arm also played its three interim
evaluations at the final gauntlet's size, about 9 s more per lane, so the
measured saving slightly understates the real one; arcs now keep every
evaluation but the last at the interim size (`--final-eval-games`). The
reference-family presets therefore run in 100-update
processes: a 400-update arc restarts three times instead of fifteen, saving
about 10% of its time, and an interrupted arc loses at most 100 updates. The
eager evaluators (`--no-vec-graph-inference`) still drift and should run in
25-update chunks (`train_reference.ps1 -ChunkSize 25`).

## Options measured and not taken

**Separate learner and snapshot graphs.** The merged graph pads every
snapshot slot to the learner's batch bucket, so a typical step computes about
1,800 rows for about 600 real ones. One merged replay costs 0.30 ms at
4 x 96 rows and 0.39 ms at 4 x 448, so a row costs at most about 0.06 us and
the padding at most about 0.05 s of GPU time per update. Replaying a learner
graph and a snapshot graph instead costs 0.44-0.57 ms per step.

**rocBLAS instead of hipBLASLt.** `TORCH_BLAS_PREFER_HIPBLASLT=0` selects
rocBLAS; `torch.backends.cuda.preferred_blas_library` has no effect on
Windows. The PPO update on synthetic data took either about 0.70 s or about
0.81 s per process, consistently within each process, under both libraries.
rocBLAS landed on the faster time in three of four processes and hipBLASLt in
two of six; their means differ by about 4%, inside that spread, so the
default library stays.

**Reduced-precision fp32 matrix multiplies.** With
`torch.backends.cuda.matmul.allow_tf32` set, hipBLASLt rejects every matrix
multiply on this GPU and PyTorch falls back, so each layer runs at about half
speed.

## Running training efficiently

- Run three lanes; `scripts/run_queue.ps1` keeps them busy with queued arcs.
- Keep CPU-heavy work, such as tests, the league or audits, off the machine
  while lanes run.
- Launch through Task Scheduler with the default affinity, as for any
  training run.
- With three lanes the GPU is the limit, and the PPO update is about three
  quarters of each update's GPU time. Lower-precision training (bf16) is the
  remaining large lever; it changes numerics and would need its own
  three-seed validation.
