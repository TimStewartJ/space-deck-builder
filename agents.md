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
python -m src train
```

> **Defaults live in `src/config.py`** (see `PPOConfig`, `RunConfig`, `DeviceConfig`, `ModelConfig`). Treat that file as the single source of truth — do not duplicate default values here or in `README.md`. Override any default on the CLI with its matching flag; run `python -m src train --help` for the full flag list.

### Self-Play Training
```bash
python -m src train --self-play
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

**Available opponent types:** `random`, `heuristic`, `simple`.

When `--self-play` is combined with `--opponents`, `--self-play-ratio` controls the split: e.g., `--self-play-ratio 0.3` means 30% of training games use past PPO snapshots and 70% use the fixed opponent pool. Snapshots are capped at 10 to bound memory. **Evaluation** runs separately against each configured opponent type and reports per-type win rates.

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
1. N games run simultaneously (default: 64 concurrent)
2. Each loop iteration advances all games past opponent moves
3. All pending PPO decisions are collected and batch-encoded
4. Single GPU forward pass for the entire batch
5. Actions distributed back to each game

### Key Files
- `src/ppo/ppo_trainer.py` — Main training loop
- `src/ppo/batch_runner.py` — BatchRunner (concurrent games + batched inference)
- `src/ppo/ppo_actor_critic.py` — Actor-Critic neural network
- `src/ai/ppo_agent.py` — PPO agent (rollout buffers, GAE, PPO update)
- `src/encoding/state_encoder.py` — Game state → tensor encoding
- `src/encoding/action_encoder.py` — Action ↔ index mapping

## Benchmarking

```bash
python -m src benchmark
```

## Evaluation / Simulation

```bash
python -m src simulate --model models/ppo_agent_XXXX.pth --games 100
```

## Tips

- **First update is slow** due to GPU/ROCm warmup. Subsequent updates are 3-5x faster.
- **Entropy collapse** (entropy dropping below 2.0) means the policy is becoming too deterministic too fast. Raise `--entropy` or lower `--lr`.
- **`--simulation-device cuda`** is faster than `cpu` because the BatchRunner does batched inference on GPU, eliminating CPU↔GPU transfer per batch.
- **Checkpoints** are saved after every update to `models/ppo_agent_{timestamp}_upd{N}_wins{W}.pth`.
- **Eval is expensive** — raise `--eval-every` for long training runs. Eval always runs on the last update regardless.
