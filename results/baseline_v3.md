# Baseline v3 — fresh-from-scratch, mlp/sum, 200 updates × 3 seeds

**Date:** 2026-04-22
**Code at:** `26ea81a` (will be re-tagged `v0.1-baseline`)
**Protocol:** [`docs/eval_protocol.md`](../docs/eval_protocol.md)

This is the reference baseline that every later post and ablation
compares against. It is intentionally the simplest possible thing:
default config, three seeds, trained from scratch, gauntleted with the
fixed inference server.

## Configuration

Everything from `src/config.py` defaults at HEAD `26ea81a`. The
relevant knobs:

| Group | Key | Value |
|---|---|---|
| Model | `actor_type`, `pool_type` | `mlp`, `sum` |
| Model | `card_emb_dim` | 32 |
| Model | `trunk_hidden_sizes` | [256, 256] |
| Model | `actor_head_sizes`, `critic_head_sizes` | [128], [128] |
| PPO | `lr` (cosine to `lr_end`) | 3e-4 → 1e-5 |
| PPO | `clip_eps`, `entropy_coef` | 0.2, 0.025 |
| PPO | `epochs`, `batch_size` | 4, 8192 |
| Run | `updates`, `episodes` | 200, 16000 |
| Run | `self_play`, `self_play_ratio` | true, 0.5 (linear ramp from 0.0) |
| Run | `opponents` (training pool) | `random` |

Each seed was launched with `python -m src train --seed N`, fully
detached, sequential. Per-seed wall time: ~100 minutes on AMD RX 9070
(ROCm 7.2).

## Results — gauntlet (5000 games per opponent type, seed 0 for game RNG)

Win rate, agent vs fixed opponent. Mean is across the three training
seeds; the spread is the seed-to-seed standard deviation.

| Opponent | Seed 0 | Seed 1 | Seed 2 | Mean | Spread (σ) |
|---|---|---|---|---|---|
| `random` | 99.78% | 99.92% | 99.92% | **99.87%** | 0.07pp |
| `heuristic` | 67.16% | 67.94% | 68.20% | **67.77%** | 0.44pp |
| `simple` | 96.16% | 96.40% | 96.22% | **96.26%** | 0.10pp |
| **Overall** | 87.70% | 88.09% | 88.11% | **87.97%** | 0.19pp |

The seed spread on every metric is small enough that any future
ablation moving a metric by more than ~1pp can be plausibly called a
real effect.

## Checkpoints

| Seed | Final checkpoint |
|---|---|
| 0 | `models/ppo_agent_0421_1915_upd200_wins3193.pth` |
| 1 | `models/ppo_agent_0421_2055_upd200_wins3197.pth` |
| 2 | `models/ppo_agent_0421_2237_upd200_wins3198.pth` |

## Reproducing

```powershell
# Train all three (sequential, ~5h total on RX 9070):
foreach ($s in 0,1,2) {
    .venv\Scripts\python.exe -m src train --seed $s
}

# Gauntlet each one:
foreach ($pair in @(@(0,'<seed-0-checkpoint>'),
                    @(1,'<seed-1-checkpoint>'),
                    @(2,'<seed-2-checkpoint>'))) {
    .venv\Scripts\python.exe -m src eval `
        --model $pair[1] `
        --opponents random,heuristic,simple `
        --games 15000 --num-workers 16 --seed $pair[0]
}
```

## Context — historic upd200 checkpoints (resumed)

For reference, three pre-`v0.1-baseline` upd200 checkpoints exist in
the repo. All three were saved during runs that used `--resume` on
top of an upd175 parent (which was itself trained in a separate run).
Total compute is still 200 PPO updates, but the resumed runs inherited
a mature self-play snapshot pool, optimizer state, and partially-
trained weights from their parent.

| Checkpoint | Architecture | vs heuristic |
|---|---|---|
| `0419_1902_upd200` (resumed) | mlp/sum | 80% |
| `sum_mlp_lr_fix_upd200` (resumed) | mlp/sum | 79% |
| `0418_0749_upd200` (resumed) | attention/attention | 84% |
| **v0.1 baselines (from scratch)** | mlp/sum | **67–68%** |

The gap between "200 updates from scratch" and "200 updates via
resume from a mature parent" — same total compute, ~12pp difference —
is large enough to flag as a real curriculum effect rather than noise.
The most likely cause is snapshot-pool warm-start: a from-scratch run
spends its early updates training against weak past selves before the
pool matures, while a resumed run starts with already-strong opponents.

This is a research thread for a future post, not a defect in the
baseline. The v0.1 baseline is deliberately the cleanest, most
reproducible thing — `git checkout v0.1-baseline`, `python -m src
train --seed N`, exact numbers.

## Cross-play Elo (secondary metric)

Round-robin tournament of the 3 baselines + 3 fixed opponents, 1000
games per pairing, alternating first-player seat per pairing. 15
pairings × 1000 games = 15000 games total, ~70s wall time.

| Rank | Player | Elo | W / L | Win % |
|---|---|---|---|---|
| 1 | baseline seed 1 (`0421_2055_upd200`) | **1016** | 3747 / 1253 | 74.9% |
| 2 | baseline seed 0 (`0421_1915_upd200`) | **1000** | 3653 / 1347 | 73.1% |
| 3 | baseline seed 2 (`0421_2237_upd200`) | **985**  | 3562 / 1438 | 71.2% |
| 4 | `heuristic` | 844 | 2739 / 2261 | 54.8% |
| 5 | `simple`    | 502 | 1237 / 3763 | 24.7% |
| 6 | `random`    | -22 |   62 / 4938 |  1.2% |

Head-to-head between baselines (1000 games each):

| Pairing | Seed-A wins |
|---|---|
| seed 0 vs seed 1 | 460 / 1000 (46%) |
| seed 0 vs seed 2 | 520 / 1000 (52%) |
| seed 1 vs seed 2 | 536 / 1000 (54%) |

The three baselines are within 31 Elo of each other and all three
inter-baseline pairings are within ~5pp of 50/50, so they are
essentially indistinguishable as players. Each baseline is ~150 Elo
above `heuristic`, validating that the gauntlet-primary ordering
(random « simple « heuristic) matches the cross-play ordering.

## Notes on the ROCm cross-stream sync bug

These numbers were produced *after* commit `26ea81a` (see
[`docs/decisions/0001-drop-multistream-inference.md`](../docs/decisions/0001-drop-multistream-inference.md)).
The pre-fix InferenceServer silently no-op'd cross-stream
synchronization on ROCm, causing the standalone eval CLI to report
~1% win rate vs heuristic for a checkpoint that scored ~76% via Elo.
The historic upd200 numbers in the table above were re-evaluated with
the fixed server, so all rows are directly comparable.
