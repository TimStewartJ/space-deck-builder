# Training & Inference Guide

## Quick Start

### Prerequisites
- Python 3.11+ with `uv` package manager
- AMD GPU (RX 9070/9000 series): Install ROCm PyTorch via TheRock
- NVIDIA GPU: Use the CUDA index in pyproject.toml
- CPU-only: Works but slower

### GPU Setup
```bash
# Install base dependencies (CPU torch by default):
uv sync

# Then install GPU-accelerated PyTorch for your hardware:
python scripts/setup_gpu.py rocm     # AMD GPUs (RX 7000/9000 series)
python scripts/setup_gpu.py cuda     # NVIDIA GPUs
python scripts/setup_gpu.py cpu      # CPU-only (already installed by uv sync)
python scripts/setup_gpu.py detect   # Auto-detect GPU and install
```

> **Note:** `uv sync` reinstalls CPU torch from the lockfile. Rerun `setup_gpu.py` after any `uv sync`.

Verify:
```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # e.g., AMD Radeon RX 9070 or NVIDIA GeForce RTX 4090
```

## Training

### Basic Training Run
```bash
python -m src.ppo.ppo_trainer \
    --updates 50 \
    --episodes 128 \
    --device cuda \
    --main-device cuda \
    --simulation-device cuda \
    --eval-every 5
```

### Recommended Settings

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `--episodes` | 1024 | 128-256 | Episodes per update. More = more data per gradient step |
| `--updates` | 4 | 50-200 | Number of PPO updates. More = longer training |
| `--device` | cuda | cuda | Device for initial model placement |
| `--main-device` | cuda | cuda | Device for PPO gradient updates |
| `--simulation-device` | cpu | cuda | Device for BatchRunner inference. **Use cuda for best performance** |
| `--lr` | 3e-4 | 3e-4 | Learning rate. Lower (1e-4) for stability |
| `--gamma` | 0.995 | 0.995 | Discount factor. High because games are long (~150 steps) |
| `--lam` | 0.99 | 0.99 | GAE lambda |
| `--clip-eps` | 0.3 | 0.2-0.3 | PPO clip range |
| `--entropy` | 0.025 | 0.01-0.05 | Entropy bonus. Higher = more exploration |
| `--epochs` | 4 | 4 | PPO epochs per update |
| `--batch-size` | 1024 | 1024-2048 | Mini-batch size for PPO updates |
| `--eval-every` | 5 | 5-10 | Evaluate every N updates (saves ~80% eval time) |
| `--eval-games` | 100 | 50-100 | Games per evaluation round |

### Self-Play Training
```bash
python -m src.ppo.ppo_trainer \
    --updates 100 \
    --episodes 128 \
    --device cuda \
    --main-device cuda \
    --simulation-device cuda \
    --self-play \
    --eval-every 10
```

Self-play trains against randomly selected past checkpoints. RandomAgent stays in the opponent pool to prevent catastrophic forgetting. Without `--self-play`, the agent trains only against RandomAgent.

### Mixed-Opponent Training
```bash
# Equal mix of random and heuristic opponents:
python -m src train --opponents random,heuristic --updates 50 --episodes 128

# Weighted mix (60% random, 30% heuristic, 10% simple):
python -m src train --opponents random:0.6,heuristic:0.3,simple:0.1

# Mixed opponents with self-play (50% of games use PPO snapshots):
python -m src train --opponents random,heuristic --self-play --self-play-ratio 0.5
```

**Available opponent types:** `random`, `heuristic`, `simple`

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--opponents` | `random` | Comma-separated opponent types with optional weights |
| `--self-play-ratio` | `0.5` | Fraction of games using PPO snapshots vs fixed opponents (only when `--self-play` is active) |

When `--self-play` is combined with `--opponents`, the self-play ratio controls the split: e.g., `--self-play-ratio 0.3` means 30% of training games use past PPO snapshots and 70% use the fixed opponent pool. Snapshots are capped at 10 to bound memory.

**Evaluation** runs separately against each configured opponent type and reports per-type win rates.

### Loading a Pretrained Model
```bash
# Load a specific checkpoint:
python -m src.ppo.ppo_trainer --model-path models/ppo_agent_0415_0200_upd10_wins65.pth ...

# Auto-load the latest checkpoint:
python -m src.ppo.ppo_trainer --load-latest-model ...
```

## Architecture & Performance

### BatchRunner
Training uses `BatchRunner`, which runs N games concurrently with batched GPU inference instead of per-step forward passes. This provides a **~14x speedup** over the original sequential architecture.

**How it works:**
1. N games run simultaneously (default: 64 concurrent)
2. Each loop iteration advances all games past opponent moves
3. All pending PPO decisions are collected and batch-encoded
4. Single GPU forward pass for the entire batch
5. Actions distributed back to each game

### Performance Expectations

| Configuration | Episodes/sec | 128 episodes |
|---------------|-------------|--------------|
| CPU sequential (old) | ~0.5 | ~235s |
| GPU BatchRunner | ~7-9 | ~15-18s |

### Key Files
- `src/ppo/ppo_trainer.py` — Main training loop
- `src/ppo/batch_runner.py` — BatchRunner (concurrent games + batched inference)
- `src/ppo/ppo_actor_critic.py` — Actor-Critic neural network
- `src/ai/ppo_agent.py` — PPO agent (rollout buffers, GAE, PPO update)
- `src/encoding/state_encoder.py` — Game state → tensor encoding
- `src/encoding/action_encoder.py` — Action ↔ index mapping

## Benchmarking

```bash
# Compare all modes:
python -m scripts.benchmark --episodes 128 --device cuda --mode both

# Batched only:
python -m scripts.benchmark --episodes 128 --device cuda --mode batched
```

## Evaluation / Simulation

```bash
# Run saved model in simulation:
python -m src.ppo.ppo_simulate --model models/ppo_agent_XXXX.pth --games 100
```

## Tips

- **First update is slow** due to GPU/ROCm warmup. Subsequent updates are 3-5x faster.
- **Entropy collapse** (entropy dropping below 2.0) means the policy is becoming too deterministic too fast. Increase `--entropy` or decrease `--lr`.
- **`--simulation-device cuda`** is faster than `cpu` because the BatchRunner does batched inference on GPU, eliminating CPU↔GPU transfer per batch.
- **Checkpoints** are saved after every update to `models/ppo_agent_{timestamp}_upd{N}_wins{W}.pth`.
- **Eval is expensive** — use `--eval-every 5` or higher for long training runs. Eval always runs on the last update regardless.
