# 0017 - Graph-replayed rollout inference and device-side PPO bookkeeping

**Status:** Accepted on `perf/training-throughput` (2026-10-30)

## Context

After decision 0009 a reference-recipe update (2,000 episodes, three active
snapshots) took 9.6 s on one training lane. Resumed from update 200 of a
phase 1 control run and pinned to eight logical performance-core CPUs, ten
warm updates of `master` (`2fb71e1`) measured:

| Part | Time per update | Where it went |
|---|---:|---|
| Rollout | 6.26 s | about 650 learner calls at 2.9 ms and 650 snapshot calls at 3.1 ms, for a few hundred rows each |
| PPO update | 3.18 s | about 13 host synchronizations per 8,192-row minibatch: seven `.item()` calls, four non-finite checks and two inside `Categorical` argument validation |
| Interim evaluation | about 23 s per 25 updates | 3,200 object-engine games with 16 worker processes |

The GPU did almost no arithmetic in those calls. Measurements on this machine
(Windows, ROCm 7.13, RX 9070, torch 2.11) show why:

- an eager forward-and-sample of the policy costs about 2.5-3 ms however few
  rows it has;
- CUDA/HIP graphs work. Replaying one costs about 40 us of host time plus
  about 3 us per elementwise node and about 38 us per GEMM node;
- asynchronous copies cost 2-10 us of host time each;
- Triton is not available, so `torch.compile` cannot fuse kernels.

## Decision

**Fold the input layer.** Card and zone embeddings, presence-weighted sum
pooling and the first trunk layer are linear in the observation, so
`PPOActorCritic` computes them as one matrix over the raw 505-value
observation (`folded_input_weight`), rebuilt from the parameters on every
forward pass. It is the same function, gradients included, without the
`[batch, zones, cards, emb]` token cube. Parameters and checkpoints are
unchanged; architectures with token features or attention pooling keep the
unfolded path.

**Replay rollout inference from graphs** (`src/vector_engine/graphed_policy.py`).
Each decision step replays one graph that evaluates the learner and every
active snapshot (or frozen opponent) with the same stacked kernels:

- weights are copied into static stacked tensors (learner in slot 0), so
  loading other snapshots never invalidates a graph; the actor and critic
  heads run as one block matrix;
- graphs are captured per batch-size bucket (four per doubling) the first time
  a bucket is needed and live for the process;
- the engine writes observations straight into pinned staging memory, and the
  CPU stages masks as `suppress_redundant_end_turn` makes them;
- a single categorical draw uses exponential noise and `argmax`, the algorithm
  `torch.multinomial` uses for one sample, without its device-side input
  checks;
- the learner's decisions, masks, log-probabilities and values are written
  inside the graph into device storage that becomes the PPO batch;
- fixed opponents choose on the CPU while the GPU evaluates the policies.

The vector runner persists across updates (`VectorBatchRunner.reconfigure`),
keeping its compiled rules and graphs. Games are dealt with whole-population
permutations instead of a per-game Python loop. Graphed inference is used for
`--engine vec` on a CUDA device when every model is the sum-pooled MLP
architecture; `--no-vec-graph-inference` and every other case use the eager
evaluators of decision 0009.

**Keep PPO bookkeeping on the device.** Minibatches are shuffled with a device
permutation; losses, ratios, KL, clip fraction, entropy and gradient norms
accumulate in a device tensor read once per update; `Categorical` skips
argument validation; Adam runs fused. A minibatch whose logits, loss or
gradient norm is not finite sets a device flag that the fused optimizer reads
as `found_inf`: it skips that step and every later one, leaving weights and
optimizer state exactly where the previous early return left them, and the
update reports NaN as before. Optimizer state saved by the unfused optimizer
loads into the fused one.

**Evaluate on the vector engine during training.** With graphed inference
available, interim and final in-training evaluations play the same games per
opponent type with the same seat alternation in one vector population
against the scalar-equivalent vector opponents of decision 0006. The `eval`
command is unchanged.

**Run presets in 100-update processes.** Graph replays do not slow as a
process ages (three lanes kept their speed through 400 updates in one
process), and each fresh process costs about 10 s. The reference-family
presets now restart every 100 updates instead of every 25; processes still
hand over optimizer, scheduler, pool and RNG state, and chunk ends still fall
on the evaluation cadence. Decision 0009's 25-update limit still applies to
the eager evaluators.

## Rejected alternatives

**Graph-captured PPO minibatches.** Capturing forward, loss, backward and
clipping per minibatch gave 0.72 s per update against 0.68 s eager: the update
is bound by GPU arithmetic (about 4.8 ms per minibatch, mostly the network's
backward GEMMs), not by launches.

**One graph per policy.** Separate learner and snapshot graphs replay twice
per decision step and double the GEMM nodes, at about 38 us of host time each.

**`torch.multinomial` inside the graph.** Its input validation adds about ten
nodes; the same exponential-noise draw costs about 43 us less per replay.

**Writing rollout storage with copies after each replay.** A device-to-device
copy outside the graph costs about 30 us of host time, more than the
`index_copy` node it replaces.

**Separate learner and snapshot graphs sized to their own rows.** Padding
snapshot slots to the learner's bucket costs at most about 0.05 s of GPU time
per update, while replaying two graphs per step costs 0.44-0.57 ms of host
time against 0.30-0.39 ms for one.

**rocBLAS.** It made the PPO update faster in some processes, but the update
time is bimodal per process under both libraries and the means differ by
about 4%, inside that spread. On Windows it can only be selected through the
`TORCH_BLAS_PREFER_HIPBLASLT` environment variable.

**Reduced-precision fp32 (`allow_tf32`).** hipBLASLt rejects every such
matrix multiply on this GPU, and the fallback halves layer speed.

**Lower precision.** bf16 matrix multiplies would roughly halve the PPO
update, but they change training numerics and would need their own
multi-seed validation.

## Consequences

- Warm single-lane updates take 2.17 s instead of 9.64 s (rollout 1.23 s,
  PPO 0.87 s), measured on the same checkpoint, updates and CPUs.
- An interim evaluation of 3,200 games takes about 1.2 s instead of about
  23 s; the locked 15,000-game final gauntlet takes about 4 s.
- A complete reference run trained from scratch on this code learns the same
  curves as the published seed 0 and lands inside the published seeds'
  spread in the final gauntlet, the held-out league and cross-play Elo
  ([`results/training_throughput.md`](../../results/training_throughput.md)).
- Random streams differ from earlier code: graph replays draw from the
  device generator's graph-safe offsets, minibatches are shuffled on the
  device and games are dealt differently. Sampling distributions are
  unchanged, so results remain comparable across seeds but are not
  bit-for-bit reproductions of earlier runs.
- The PPO update now dominates GPU time, so lanes running side by side
  contend for the GPU rather than the CPU. Three lanes saturate the project
  machine's RX 9070 at about one update per second in total; more lanes add
  nothing, and five oversubscribe the CPUs.
- Graph warm-ups share one side stream per device. hipBLASLt allocates a
  workspace for every stream it sees and never frees it, so a new stream per
  capture leaked about 2.2 GB per lane.
- A 400-update arc restarts three times instead of fifteen, about 10% less
  time. An arc started under the old 25-update default resumes only with
  `train_reference.ps1 -ChunkSize 25`, because the script rejects a changed
  invocation.
