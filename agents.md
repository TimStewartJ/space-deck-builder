# Training & Inference Guide

## Quick Start

### Prerequisites
- Python 3.11+ with `uv` package manager
- AMD GPU (RX 9070/9000 series): Install ROCm PyTorch via TheRock
- NVIDIA GPU: Use the CUDA index in pyproject.toml
- CPU-only: Works but slower

### GPU Setup

`torch` lives only in optional-dependency extras, never in base `dependencies`.
Pick exactly one backend extra:

```bash
uv sync --extra rocm    # AMD GPUs (ROCm via TheRock)
uv sync --extra cuda    # NVIDIA GPUs (CUDA 12.8)
uv sync --extra cpu     # CPU-only

# Or let the helper detect your hardware and pick the extra for you:
python scripts/setup_gpu.py detect
```

> **Never run bare `uv sync`.** Because `torch` is only declared in extras,
> a bare `uv sync` makes the venv match a torch-free dependency set and
> **uninstalls torch entirely**. This is correct uv behavior, not a bug —
> `sync` is declarative. Always pass `--extra <backend>`.

`scripts/setup_gpu.py` is a thin wrapper around `uv sync --extra <backend>`
plus a verification step; it is convenience, not a separate install path.

Verify:
```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # e.g., AMD Radeon RX 9070 or NVIDIA GeForce RTX 4090
```

> **Pinned torch on Windows/ROCm.** `pyproject.toml` and the lockfile pin
> `torch 2.11.0+rocm7.13.0a20260408` and its matching ROCm packages. AMD now
> publishes newer multi-arch Windows nightlies, but the tested July 28
> torch 2.12/ROCm 7.15/gfx1201 stack regressed the 30-update workload by 67%
> and did not remove process-age degradation. Upgrade the complete stack only
> through an isolated A/B experiment; refresh unrelated dependencies with
> `uv lock --upgrade-package <name>`.

### Running code

Use the venv interpreter directly so nothing re-triggers a sync:

```bash
.venv\Scripts\python.exe -m src <command>
```

`uv run` (without `sync`) is also safe. Each git worktree has its own
`.venv` (~4.8 GB with torch), so a fresh worktree needs its own
`uv sync --extra rocm` before it can run anything.

## Training

### Basic Training Run
```bash
python -m src train
```

> **Defaults live in `src/config.py`** (see `PPOConfig`, `RunConfig`, `DeviceConfig`, `ModelConfig`). Treat that file as the single source of truth — do not duplicate default values here or in `README.md`. Override any default on the CLI with its matching flag; run `python -m src train --help` for the full flag list.

### Reference Recipe
The strongest published recipe is a named preset in `TRAINING_PRESETS`
(`src/config.py`). Train one seed end to end with:

```powershell
.\scripts\train_reference.ps1 -Seed 0
```

The script registers a Task Scheduler launch, trains the first chunk from
scratch, continues in fresh 25-update processes with RNG/optimizer/pool
continuity, and finishes with the locked exact-seat gauntlet. Re-running the
same command resumes the arc. `python -m src train --preset reference-selfplay`
applies the same values in a single process; explicit flags still override
the preset.

### Launching Long Runs
Launch training, evaluation, tournaments, and benchmarks outside the agent
process tree. A child of the agent runtime measured ~3.7x slower than the same
run started through Task Scheduler, and `Start-Process` does not escape that
execution context. Use `scripts\launch_training.ps1` (`-TrainArgs` for
`python -m src train`, `-Command` for any other command with `{python}` as the
interpreter placeholder) and stop runs with `scripts\stop_training.ps1`.

### Self-Play Training
```bash
python -m src train --self-play
```

Self-play trains against randomly selected past checkpoints while keeping the fixed opponent pool available to prevent catastrophic forgetting. Without `--self-play`, the agent trains only against the fixed opponent pool configured in `RunConfig.opponents`.

### Mixed-Opponent Training
```bash
# Equal mix of random and heuristic opponents:
python -m src train --opponents random,heuristic --updates 50 --episodes 128

# Weighted mix (60% random, 30% heuristic, 10% simple):
python -m src train --opponents random:0.6,heuristic:0.3,simple:0.1

# Mixed opponents with self-play (50% of games use PPO snapshots):
python -m src train --opponents random,heuristic --self-play --self-play-ratio 0.5
```

**Available opponent types:** `random`, `heuristic`, `simple`.

When `--self-play` is combined with `--opponents`, `--self-play-ratio` controls the split: e.g., `--self-play-ratio 0.3` means 30% of training games use past PPO snapshots and 70% use the fixed opponent pool. The snapshot pool is capped (`snapshot_cap`, default 10) and evicts on a log-spaced age ladder, so the pool keeps a spread of old and recent selves instead of collapsing to near-clones of the current policy. `--pfsp {uniform,hard,variance}` changes how snapshots are sampled: `hard` favors opponents the agent loses to, `variance` favors opponents near a 50% win rate. **Evaluation** runs separately against each configured opponent type and reports per-type win rates.

### Training Against Frozen Checkpoints
```bash
# Every training game against selfplay_seed0 playing its highest-scoring move:
python -m src train --engine vec --model-path <start.pth> \
    --frozen-opponents results/optimized_selfplay_baseline/checkpoints/selfplay_seed0.pth \
    --frozen-opponent-ratio 1 --frozen-opponent-temperature 0
```

`--frozen-opponents` loads checkpoints into the opponent pool at the start of
every process, fresh or resumed; they never learn and are never written into a
checkpoint's pool manifest. `--frozen-opponent-ratio` is the share of training
games against them (the rest use `--opponents`), and
`--frozen-opponent-temperature 0` makes them play their highest-scoring legal
action. It requires `--engine vec` and excludes `--self-play`. The
`exploit-selfplay-seed0` preset trains an exploiter of the published champion
this way, and `continue-selfplay-seed0` is its control
(`docs/decisions/0013-strength-yardstick.md`). `exploit-ladder-random` trains
the fallback exploiter from random weights against the champion and five
earlier checkpoints of its training run
(`docs/decisions/0014-ladder-exploiter.md`); those five are published as
release assets and go in `results/strength_yardstick/ladder/`.

### Reproducibility

```bash
python -m src train --seed 0        # seeds random, numpy, and torch
```

Per `docs/eval_protocol.md`, any run that produces a reported checkpoint is
repeated with seeds `0`, `1`, `2`. ROCm is not bitwise-deterministic, so
multi-seed averaging — not per-seed bit reproduction — is what makes results
comparable.

### Loading a Pretrained Model
```bash
# Resume from a specific checkpoint:
python -m src train --resume models/ppo_agent_XXXX.pth

# Auto-load the latest checkpoint:
python -m src train --load-latest-model
```

## Architecture & Performance

### Rollout
With `--engine object` (the `RunConfig` default), training runs `num_workers` simulation processes against a single batched inference server that owns the GPU. The same multiprocess runner powers `eval` and `elo`.

**How it works:**
1. `num_concurrent` games run simultaneously **per worker** — it defaults to
   `episodes // num_workers` (800 at the shipped defaults), so with
   `num_workers=20` all 16,000 episodes of an update are in flight at once
2. Each worker splits its slots into pipeline groups and keeps one request per
   group in flight, so it encodes one group while the other is on the GPU
3. Encoded states go straight into a shared-memory block; only coordinates
   travel over the queues
4. The server coalesces requests across workers into one forward pass per
   model, then writes results back into shared memory in place
5. Workers read results in place and apply the sampled actions

Shipping arrays through `mp.Queue` instead cost a pickle, pipe write, unpickle
and staging copy per decision — roughly 4.7 GB per update, all serialised
through the single server thread.

### Compiler-backed vector engine
`src/vector_engine/` compiles all 49 canonical cards from the existing `Effect`
trees into fixed-width opcode tables and runs the rules in one Numba SPMD
process, including pending target choices and continuation effects.

Correctness is differential: scalar and vector legal masks/observations match
before every action, and complete canonical state matches afterward. Gate 2
measured 5.88M engine+mask+observation decisions/s and 587k/s with the actual
GPU policy, so rollouts are policy-bound. `train --engine vec` supports the
fixed Random/Heuristic/Simple opponents, weighted opponent specifications, and
PFSP snapshot self-play with stacked actor-only snapshot inference; the
reference preset trains on it. Object remains the `RunConfig` default for
compatibility.

### Information-set search support
`src/search/information_set.py` projects a live scalar decision into only the
acting player's public knowledge and samples seeded hidden hand, deck, and
trade-deck arrangements from public ownership ledgers without reading the true
hidden allocation. `src/search/vector_root.py` writes the same information set
straight into a vector root from public counts, with no deep copies, and
`src/search/turn_search.py` searches many games' decisions together in one
vector population, guided by the policy and critic.
`src/search/batched_games.py` plays many games concurrently with batched
policy and search decisions, and `python -m src search-eval` measures search
uplift against the same checkpoint without search (ADR 0010 and ADR 0011).
`python -m src search-selfplay` records search-versus-search games as training
shards with completed-Q improved policies and outcomes, and
`python -m src distill` trains a checkpoint on them (ADR 0012).
Launch large `search-eval` runs and `scripts/bench_turn_search.py` through
`scripts/launch_training.ps1 -Command`.

### Diagnosing a slow rollout
`scripts/bench_rollout.py` reports decisions/sec plus a per-phase breakdown of
the server (`idle drain stage h2d kernel sync dispatch`) and the workers
(`advance encode wait apply`, with busy/blocked percentages). Start there
rather than guessing:

```bash
python scripts/bench_rollout.py --updates 3
```

Read it this way:
- **workers `blocked` high** → the server is the bottleneck; look at its phases
- **server `idle` high** → workers are the bottleneck; look at `encode`
- **`stage` or `drain` high** → transport regression
- **`kernel` a small fraction of server wall** → the GPU is not the limit

### Key Files
- `src/ppo/ppo_trainer.py` — Main training loop
- `src/ppo/batch_runner.py` — Single-process runner (`num_workers <= 1`)
- `src/ppo/mp_batch_runner.py` — Multi-process runner, owns the worker fleet
- `src/ppo/mp_sim_worker.py` — Worker game loop, pipelined submit/collect
- `src/ppo/mp_inference_server.py` — Batched GPU inference server
- `src/ppo/shared_io.py` — Shared-memory transport for the inference round trip
- `src/vector_engine/rules.py` — Effect-to-IR compiler and scalar IR interpreter
- `src/vector_engine/state.py` — SoA state and scalar canonical conversion
- `src/vector_engine/kernels.py` — Numba rules, masks, observations, and wrapper
- `src/vector_engine/differential.py` — Per-step scalar/vector oracle
- `src/vector_engine/policy.py` — Actual PPO policy integration
- `src/vector_engine/opponents.py` — Scalar-equivalent vector fixed opponents
- `src/ppo/vector_batch_runner.py` — Vector rollout runner, including snapshot self-play
- `src/search/information_set.py` — Information-set projection and determinization
- `src/search/vector_root.py` — Search roots written from public counts
- `src/search/vector_determinization.py` — Vectorized hidden-hand dealing for search worlds
- `src/search/turn_search.py` — Batched turn-bounded rollout search with vector worlds
- `src/search/batched_games.py` — Concurrent games with batched policy and search seats
- `src/search/uplift.py` — Search-versus-baseline match harness (`search-eval`)
- `src/search/selfplay_data.py` — Search self-play training shards (`search-selfplay`)
- `src/search/distill.py` — Distillation of search targets into a checkpoint (`distill`)
- `src/ppo/opponent_pool.py` — Fixed opponents, self-play snapshots, frozen checkpoints, PFSP weighting
- `src/ppo/elo_tournament.py` — Cross-play Elo tournaments
- `src/ai/held_out_agents.py` — Held-out scripted opponents, never used in training
- `src/ppo/league.py` — Held-out league (`league`)
- `src/search/purchase_audit.py` — Rollout audit of purchase decisions (`purchase-audit`)
- `src/ppo/ppo_actor_critic.py` — Actor-Critic neural network
- `src/ai/ppo_agent.py` — PPO agent (rollout buffers, GAE, PPO update)
- `src/encoding/state_encoder.py` — Game state → tensor encoding
- `src/encoding/state_utils.py` — Zone layout (`ZONE_NAMES`) and state unpacking
- `src/encoding/action_encoder.py` — Action ↔ index mapping

## Benchmarking

```bash
python -m src benchmark
```

## Evaluation / Simulation

The `eval` and `elo` subcommands are the reporting tools defined by
[`docs/eval_protocol.md`](docs/eval_protocol.md); `simulate` and `analyze`
are for inspection rather than reported numbers.

```bash
# Locked gauntlet: 5000 games per opponent type
python -m src eval --model models/ppo_agent_XXXX.pth \
    --opponents random,heuristic,simple --games 15000 --seed 0 --num-workers 16

# Cross-play Elo across checkpoints and fixed agents
python -m src elo --checkpoints <ckpt-a> <ckpt-b> \
    --agents random,heuristic,simple --games-per-pair 1000 --num-workers 16

# Evaluate the highest-probability action instead of sampling, and write JSON
python -m src eval --model models/ppo_agent_XXXX.pth --temperature 0 \
    --games 15000 --seed 0 --num-workers 16 --output-json eval.json

# Include every checkpoint both sampled and greedy in one tournament
python -m src elo --checkpoints <ckpt-a> <ckpt-b> --checkpoint-temperatures 1,0 \
    --agents random,heuristic,simple --games-per-pair 1000 --output-json elo.json

# Watch a model play, or collect replays for behavioral analysis
python -m src simulate --model models/ppo_agent_XXXX.pth --games 100
python -m src analyze --model models/ppo_agent_XXXX.pth
```

The held-out league and the purchase audit (protocol `v0.2`, decision 0013)
measure strength against opponents no agent trains on and the win probability
purchases give away. Both journal finished work and resume when re-run; launch
large runs through `scripts/launch_training.ps1 -Command`. Each league worker
is a separate process that uses the GPU and keeps about three CPU cores busy.
On the project's machine (20 logical CPUs) six workers use all of the CPU, and
twelve finished no 4,000-game matchup in 30 minutes, so keep `--workers` at 6
and do not run the league or an audit beside training.

```bash
# Every checkpoint against the six held-out opponents, 4,000 games each
python -m src league --models <ckpt-a> <ckpt-b> --calibration --workers 6 \
    --output-json league.json

# 1,000 purchase decisions scored by complete greedy rollouts
python -m src purchase-audit --model <ckpt> --output-json audit.json
```

## Tips

- **First update is slow** due to GPU/ROCm warmup. Subsequent updates are 3-5x faster.
- **Watch entropy against the observed baseline, not a fixed threshold.** On the
  reference self-play preset a healthy run starts near `ent≈0.97`, falls to
  `ent≈0.78` by update 25, and settles near `ent≈0.72`. (The historical
  object-engine 200 × 16,000 recipe drifted from `ent≈0.90` to `ent≈0.57`.)
  Entropy falling far faster than that early on means the policy is going
  deterministic before it has explored; raise `--entropy` or lower `--lr`.
- **`kl` and `clip` tell you whether updates are actually landing.** The shipped
  PPO settings produce a small trust-region step (`kl≈0.002–0.004`,
  `clip≈0.02–0.03` at full LR); both fall toward zero as the cosine schedule
  reaches `lr_end`. Near-zero values at full LR mean the policy has stopped
  moving.
- **`ev` (explained variance)** is the critic's fit to observed returns. It
  should climb early and stay high; a sustained decline means the value
  function is losing track of the state distribution.
- **`--simulation-device cuda`** is faster than `cpu` because the BatchRunner does batched inference on GPU, eliminating CPU↔GPU transfer per batch.
- **Checkpoints** are saved after every update to `models/ppo_agent_{timestamp}_upd{N}_wins{W}.pth`. Only eval updates record a real win count; others save `wins-1`.
- **Eval is expensive** — raise `--eval-every` for long training runs. Eval always runs on the last update regardless.
- **Publish checkpoints a result depends on** with `scripts/release_artifacts.py publish` rather than committing them; see `docs/eval_protocol.md` §8.
