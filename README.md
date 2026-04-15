# Space Deck Builder Simulator

A simulator and AI trainer for a space-based deck building card game, featuring PPO reinforcement learning with batched GPU inference.

## Quick Start

```bash
# Install dependencies
uv sync

# GPU setup (optional — CPU works but is slower)
python scripts/setup_gpu.py detect   # auto-detect and install GPU PyTorch

# Train a PPO agent
python -m src train --updates 50 --episodes 128

# Simulate games with a trained model
python -m src simulate --games 100

# Benchmark training throughput
python -m src benchmark --episodes 128
```

## Unified CLI

All commands go through `python -m src <command>`:

| Command | Description |
|---------|-------------|
| `train` | Run PPO training with configurable hyperparameters |
| `simulate` | Run a trained model against opponents |
| `benchmark` | Compare sequential vs batched training throughput |

Run `python -m src <command> --help` for full option details.

### Training Examples

```bash
# Basic training
python -m src train --updates 50 --episodes 128 --device cuda

# Self-play training
python -m src train --updates 100 --episodes 128 --self-play

# Mixed opponents with custom weights
python -m src train --opponents random:0.6,heuristic:0.4

# Resume from a checkpoint
python -m src train --load-latest-model --updates 50
```

### Key Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--episodes` | 1024 | Episodes per update |
| `--updates` | 4 | Number of PPO updates |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.995 | Discount factor |
| `--clip-eps` | 0.3 | PPO clip range |
| `--entropy` | 0.025 | Entropy bonus coefficient |
| `--self-play` | off | Train against past snapshots |
| `--opponents` | random | Opponent mix (random, heuristic, simple) |
| `--main-device` | cuda | Device for gradient updates |
| `--simulation-device` | cpu | Device for episode simulation |
| `--eval-every` | 5 | Evaluate every N updates |

## Configuration System

Training parameters are defined as centralized dataclasses in `src/config.py`:

- **`GameConfig`** — game rules (starting health, hand sizes, trade row size)
- **`DataConfig`** — card data paths and set filtering
- **`ModelConfig`** — neural network architecture (embedding dims, hidden sizes)
- **`PPOConfig`** — PPO hyperparameters (lr, gamma, clip epsilon, etc.)
- **`RunConfig`** — training topology (episodes, updates, concurrency, opponents)
- **`DeviceConfig`** — GPU/CPU device placement
- **`SimConfig`** — simulation settings

CLI arguments map directly to these config objects. Checkpoints save config metadata for reproducibility.

## Architecture

### BatchRunner

Training uses `BatchRunner` for concurrent game simulation with batched GPU inference:

1. N games run simultaneously (default: 64 concurrent)
2. Opponent moves are advanced in bulk
3. All pending PPO decisions are batch-encoded
4. Single GPU forward pass for the entire batch
5. Actions distributed back to each game

This provides ~14x speedup over sequential per-game inference.

### Project Structure

```
src/
├── __main__.py          # Unified CLI entrypoint
├── config.py            # Centralized config dataclasses
├── ai/                  # Agent implementations (PPO, Random, Heuristic)
├── cards/               # Card loading, effects, and parsing
├── encoding/            # State and action tensor encoding
├── engine/              # Game engine (Game, Player, actions)
├── ppo/                 # PPO training (trainer, BatchRunner, actor-critic)
└── utils/               # Logging utilities
scripts/
├── benchmark.py         # Training throughput benchmarking
└── setup_gpu.py         # GPU PyTorch installation helper
data/
└── cards.csv            # Card definitions (Core Set)
```

## Disclaimers

### Third-Party Content (Card Data)

This simulator utilizes card information publicly made available by Wise Wizard Games via a Google Spreadsheet. This spreadsheet content is copyrighted by Wise Wizard Games (© 2021-2024 Wise Wizard Games). The maintainers of this simulator project do not claim ownership of this data and provide no warranty regarding its accuracy, completeness, or timeliness. Any discrepancies or errors in the card data are solely the responsibility of the original source. The maintainers of this project disclaim all liability related to the use of this third-party data.

### Intellectual Property

Star Realms, its logo, card names, artwork, and related elements are trademarks and copyrights of Wise Wizard Games. Use of this intellectual property in this simulator is for descriptive, non-commercial purposes only.
