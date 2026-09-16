# Space Deck Builder Simulator

A simulator and AI trainer for a space-based deck building card game, featuring PPO reinforcement learning with batched GPU inference.

## Quick Start

```bash
# Install dependencies (CPU-only torch by default)
uv sync --extra cpu

# GPU setup — pick your backend:
uv sync --extra rocm    # AMD GPUs (ROCm via TheRock)
uv sync --extra cuda    # NVIDIA GPUs (CUDA 12.8)
# Or auto-detect:
python scripts/setup_gpu.py detect

# Train a PPO agent (defaults live in src/config.py)
uv run --extra rocm python -m src train

# Simulate games with a trained model
uv run --extra rocm python -m src simulate --games 100

# Benchmark training throughput
uv run --extra rocm python -m src benchmark
```

> **Note:** Replace `--extra rocm` with your GPU backend (`cuda` or `cpu`).
> After the initial `uv sync --extra <backend>`, plain `uv run` also works
> without clobbering your GPU torch install.

## Unified CLI

All commands go through `python -m src <command>`:

| Command | Description |
|---------|-------------|
| `train` | Run PPO training with configurable hyperparameters |
| `eval` | Evaluate a checkpoint against the fixed opponent gauntlet |
| `elo` | Cross-play Elo tournament between checkpoints and/or agents |
| `simulate` | Run a trained model against opponents |
| `analyze` | Collect replays and analyze agent behavior |
| `benchmark` | Compare sequential vs batched training throughput |

Run `python -m src <command> --help` for full option details.

`eval` and `elo` are the reporting tools defined by
[`docs/eval_protocol.md`](docs/eval_protocol.md) — that document is the
contract every result in `results/` is measured against. Both sample each
checkpoint's policy by default; `eval --temperature 0` and
`elo --checkpoint-temperatures 1,0` evaluate the highest-probability action
as a separate variant, and `--output-json` writes machine-readable results.

### Reference Training Recipe

The strongest published agent comes from the `reference-selfplay` preset in
`src/config.py` ([`results/optimized_selfplay_baseline.md`](results/optimized_selfplay_baseline.md)).
Train one seed end to end on Windows with:

```powershell
.\scripts\train_reference.ps1 -Seed 0
```

The script launches through Task Scheduler (outside the caller's process
tree), trains in fresh 25-update processes with full resume continuity, and
finishes with the locked exact-seat gauntlet. Re-running the same command
resumes an interrupted arc. On other platforms, or for a single-process run,
use `python -m src train --preset reference-selfplay --seed 0`; explicit
flags override preset values.

The published checkpoints are available from the
[`v0.3-selfplay-baseline`](https://github.com/TimStewartJ/space-deck-builder/releases/tag/v0.3-selfplay-baseline)
release:

```bash
python scripts/release_artifacts.py fetch --tag v0.3-selfplay-baseline \
    --bundle optimized_selfplay_baseline-checkpoints \
    --dest results/optimized_selfplay_baseline
```

### Training Examples

```bash
# Reference self-play recipe in one process
uv run --extra rocm python -m src train --preset reference-selfplay --seed 0

# Basic training (fixed opponents from src/config.py)
uv run --extra rocm python -m src train

# Self-play training
uv run --extra rocm python -m src train --self-play

# Self-play with PFSP (prioritize challenging snapshots)
uv run --extra rocm python -m src train --self-play --pfsp hard

# Mixed opponents with custom weights
uv run --extra rocm python -m src train --opponents random:0.6,heuristic:0.4

# Controlled legacy-style random-only training
uv run --extra rocm python -m src train --opponents random

# Experimental token-feature model path (not the default baseline)
uv run --extra rocm python -m src train --token-features

# Resume from the latest checkpoint
uv run --extra rocm python -m src train --load-latest-model

# Resume a checkpoint's self-play snapshot curriculum
uv run --extra rocm python -m src train --load-latest-model --self-play
```

### Key Training Options

Defaults are defined in `src/config.py` — see the **Configuration System** section below.
Run `python -m src train --help` for the full, authoritative flag list.

`--token-features` is experimental and disabled by default. It enables static
card metadata features in the model input path, but `mlp` / `sum` without token
features remains the baseline architecture.

## Configuration System

Training parameters are defined as centralized dataclasses in `src/config.py`:

- **`GameConfig`** — game rules (starting health, hand sizes, trade row size)
- **`DataConfig`** — card data paths and set filtering
- **`ModelConfig`** — neural network architecture (embedding dims, hidden sizes)
- **`PPOConfig`** — PPO hyperparameters (lr, gamma, clip epsilon, etc.)
- **`RunConfig`** — training topology (episodes, updates, concurrency, opponents)
- **`DeviceConfig`** — GPU/CPU device placement
- **`SimConfig`** — simulation settings
- **`TRAINING_PRESETS`** — published training recipes applied with `train --preset`

CLI arguments map directly to these config objects. Checkpoints save config metadata for reproducibility.

## Architecture

### Rollout architecture

With `--engine object`, training runs `num_workers` simulation processes
against a single batched inference server that owns the GPU. `eval` and `elo`
use the same multiprocess runner:

1. `num_concurrent` games run simultaneously **per worker** — it defaults to
   `episodes // num_workers` (800 at the shipped defaults), so with 20 workers
   all 16,000 episodes of an update are in flight at once
2. Each worker splits its game slots into pipeline groups and keeps one
   inference request per group in flight, so it encodes one group while the
   other's batch is on the GPU
3. Encoded states are written directly into a shared-memory block
   (`src/ppo/shared_io.py`); only coordinates travel over the queues
4. The server coalesces requests across workers into one forward pass per
   model, then writes actions, log-probs and values back into shared memory
5. Workers read their results in place and apply the sampled actions

Shared memory replaced sending numpy arrays through `mp.Queue`, which cost a
pickle, a pipe write, an unpickle and a staging copy per decision — about
4.7 GB per update, all funnelled through the single server thread. Masks stay
`uint8` end to end because the sampling path only compares them.

### Compiler-backed vector engine

`src/vector_engine/` is a rules-correct experimental replacement for the
worker/IPC path. It compiles all 49 canonical cards from the existing
declarative `Effect` trees to fixed-width opcode tables consumed by a Numba
SPMD engine, including pending target choices, continuation effects, ordered
top-deck placement, passive triggers, and per-copy rules for Stealth Needle.
The same versioned Rule IR drives the scalar interpreter, so the object and
vector engines no longer maintain separate effect-type implementations.

Snapshot self-play reuses the vector population. Games receive stable PFSP
snapshot assignments, and opponent rows are grouped into one batched inference
call per active snapshot model. By default, PFSP selects at most three snapshot
models per rollout so the full historical pool remains available across
updates without fragmenting every inference iteration across ten tiny batches.

The scalar engine remains the oracle: tests compare every legal mask and
505-float observation before each action, then every canonical state field
afterward. The final detached full-card 16,000-game gate measured 5.88M
decisions/s for engine + masks + observations and 587k/s with the actual PPO
policy.
Use `train --engine vec --opponents random,heuristic,simple` for the guarded
mixed-opponent path. Weighted specifications and snapshot self-play use the
same configuration as the object runner. Object remains the default. See
[`docs/decisions/0003-compiler-backed-numba-vector-engine.md`](docs/decisions/0003-compiler-backed-numba-vector-engine.md)
and [`docs/decisions/0004-complete-vector-engine-and-training-path.md`](docs/decisions/0004-complete-vector-engine-and-training-path.md)
and [`docs/decisions/0005-authoritative-rule-ir-and-core-fidelity.md`](docs/decisions/0005-authoritative-rule-ir-and-core-fidelity.md)
and [`docs/decisions/0006-vector-fixed-opponents.md`](docs/decisions/0006-vector-fixed-opponents.md)
and [`docs/decisions/0008-vector-snapshot-self-play.md`](docs/decisions/0008-vector-snapshot-self-play.md)
and [`results/vector_engine_gate.md`](results/vector_engine_gate.md).

### Information-safe search state

`src/search/information_set.py` projects a live scalar decision into only the
facts available to the acting player, then samples seeded hidden hand/deck and
trade-deck arrangements from public ownership ledgers. Determinization never
uses the simulator's true hidden allocation, preserves publicly known hand and
top-deck constraints, and returns independent scalar states that convert
directly into the vector engine for search.

### Turn-bounded search

`src/search/turn_search.py` compares the policy's highest-prior legal actions
by truncated rollouts inside the vector engine. Each candidate is applied in
worlds that re-deal only the opponent's unseen hand. The searcher's turn is
continued greedily, and each world is scored by the exact outcome or by the
negated critic value of the opponent's first decision. Sequential halving
spends a fixed rollout budget. Results depend only on public information; see
[`docs/decisions/0010-turn-bounded-search.md`](docs/decisions/0010-turn-bounded-search.md).
At 64 rollouts per decision it beats the same checkpoint playing greedily on
56.4% of paired deals (+44 Elo; [`results/search_uplift.md`](results/search_uplift.md)).

Search roots are written straight from public counts
(`src/search/vector_root.py`) without copying games, and
`TurnSearch.choose_many` searches many games' decisions in one shared vector
population, so a rollout step costs one policy evaluation for all of them.
`src/search/batched_games.py` plays hundreds of games concurrently with
batched policy and search decisions; see
[`docs/decisions/0011-batched-search-and-count-roots.md`](docs/decisions/0011-batched-search-and-count-roots.md).

```bash
# Search at several budgets against the same checkpoint playing greedily
python -m src search-eval --model results/optimized_selfplay_baseline/checkpoints/selfplay_seed0.pth \
    --budgets 0 16 64 256 --games 1000 --batch-size 1024 --output-json uplift.json
```

### Measuring rollout performance

`scripts/bench_rollout.py` runs a short training and prints decisions/sec
alongside a full per-phase breakdown of both the server (idle, drain, stage,
h2d, kernel, sync, dispatch) and the workers (advance, encode, wait, apply,
and how much of their time was spent blocked):

```bash
uv run --extra rocm python scripts/bench_rollout.py --updates 3
```

Always launch performance measurements outside automation/agent process trees.
The same rollout measured about 3.7x slower as a child of the agent runtime
despite Normal process priority. The Task Scheduler launchers under `scripts/`
avoid that inherited execution context.

For training, use `scripts/launch_training.ps1`. It also applies the measured
P-core affinity on the project's i7-12700K host, and `-Command` launches any
other workload (evaluation, tournaments, orchestration scripts) through the
same boundary. Long self-play arcs should use 25-update
`scripts/chunked_resume.ps1` processes so ROCm latency cannot accumulate inside
one Python runtime; `scripts/train_reference.ps1` combines both for the
reference recipe.

### Project Structure

```
src/
├── __main__.py          # Unified CLI entrypoint
├── config.py            # Centralized config dataclasses
├── ai/                  # Agent implementations (PPO, Random, Heuristic, Simple)
├── analysis/            # Replay collection and behavioral analysis
├── cards/               # Card loading, effects, and parsing
├── encoding/            # State and action tensor encoding
├── engine/              # Game engine (Game, Player, actions)
├── ppo/                 # PPO training (trainer, runners, shared-memory IO, opponent pool, Elo)
├── search/              # Information-set projection, turn-bounded search, uplift harness
├── ui/                  # CLI and pygame interfaces
└── utils/               # Logging utilities
docs/
├── eval_protocol.md     # The measurement contract for every result
└── decisions/           # Architecture decision records
results/                 # Experiment writeups + their committed evidence
scripts/                 # Developer tools — see scripts/README.md for the full index
tests/                   # pytest suite
data/
└── cards.csv            # Card definitions (Core Set)
```

## Disclaimers

### Third-Party Content (Card Data)

This simulator utilizes card information publicly made available by Wise Wizard Games via a Google Spreadsheet. This spreadsheet content is copyrighted by Wise Wizard Games (© 2021-2024 Wise Wizard Games). The maintainers of this simulator project do not claim ownership of this data and provide no warranty regarding its accuracy, completeness, or timeliness. Any discrepancies or errors in the card data are solely the responsibility of the original source. The maintainers of this project disclaim all liability related to the use of this third-party data.

### Intellectual Property

Star Realms, its logo, card names, artwork, and related elements are trademarks and copyrights of Wise Wizard Games. Use of this intellectual property in this simulator is for descriptive, non-commercial purposes only.
