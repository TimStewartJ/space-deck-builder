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

> **Pinned torch on Windows/ROCm.** The lockfile pins
> `torch 2.11.0+rocm7.13.0a20260408`. That is the highest torch version AMD
> publishes for `win_amd64` on the gfx120X nightly index — newer nightly
> *dates* only ship torch 2.10.0 for Windows, so "upgrading" by date is a
> downgrade. Refresh non-torch dependencies with
> `uv lock --upgrade-package <name>` rather than a blanket `uv lock --upgrade`.

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

### BatchRunner
Training uses `BatchRunner`, which runs N games concurrently with batched GPU inference instead of per-step forward passes. This provides a **~14x speedup** over the original sequential architecture.

**How it works:**
1. `num_concurrent` games run simultaneously **per worker**. It defaults to
   `episodes // num_workers` (800 at the shipped defaults), and with
   `num_workers=20` that puts all 16,000 episodes of an update in flight at once
2. Each loop iteration advances all games past opponent moves
3. All pending PPO decisions are collected and batch-encoded
4. Single GPU forward pass for the entire batch
5. Actions distributed back to each game

### Key Files
- `src/ppo/ppo_trainer.py` — Main training loop
- `src/ppo/batch_runner.py` — BatchRunner (concurrent games + batched inference)
- `src/ppo/mp_batch_runner.py` — Multi-process runner used when `num_workers > 1`
- `src/ppo/opponent_pool.py` — Fixed opponents, self-play snapshots, PFSP weighting
- `src/ppo/elo_tournament.py` — Cross-play Elo tournaments
- `src/ppo/ppo_actor_critic.py` — Actor-Critic neural network
- `src/ai/ppo_agent.py` — PPO agent (rollout buffers, GAE, PPO update)
- `src/encoding/state_encoder.py` — Game state → tensor encoding
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

# Watch a model play, or collect replays for behavioral analysis
python -m src simulate --model models/ppo_agent_XXXX.pth --games 100
python -m src analyze --model models/ppo_agent_XXXX.pth
```

## Tips

- **First update is slow** due to GPU/ROCm warmup. Subsequent updates are 3-5x faster.
- **Watch entropy against the observed baseline, not a fixed threshold.** On the
  shipped defaults a healthy 200-update run starts near `ent≈0.90` and drifts to
  `ent≈0.57`. Entropy falling far faster than that early on means the policy is
  going deterministic before it has explored; raise `--entropy` or lower `--lr`.
- **`kl` and `clip` tell you whether updates are actually landing.** The shipped
  defaults produce a small trust-region step (`kl≈0.003`, `clip≈0.03` at full
  LR). Near-zero values mean the policy has effectively stopped moving.
- **`ev` (explained variance)** is the critic's fit to observed returns. It
  should climb early and stay high; a sustained decline means the value
  function is losing track of the state distribution.
- **`--simulation-device cuda`** is faster than `cpu` because the BatchRunner does batched inference on GPU, eliminating CPU↔GPU transfer per batch.
- **Checkpoints** are saved after every update to `models/ppo_agent_{timestamp}_upd{N}_wins{W}.pth`. Only eval updates record a real win count; others save `wins-1`.
- **Eval is expensive** — raise `--eval-every` for long training runs. Eval always runs on the last update regardless.
