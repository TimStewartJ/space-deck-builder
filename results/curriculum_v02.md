# Curriculum v0.2 - fixed mixed opponents vs self-play

**Date:** 2026-04-28
**Branch:** `fix/curriculum-defaults`
**Code at training time:** `9cd76b8`
**Protocol:** [`docs/eval_protocol.md`](../docs/eval_protocol.md)

This experiment tests the curriculum change discovered after the v0.1
architecture work: recent baselines had trained against `random` only, even
though parts of the config/docs implied a broader curriculum. The v0.2 branch
makes the default training curriculum match the fixed gauntlet:

```text
python -m src train
=> opponents=random,heuristic,simple
=> self_play=False

python -m src train --self-play
=> opponents=random,heuristic,simple + PPO snapshot opponents
```

The question is whether the stable fixed-opponent mix is enough for the next
default, or whether self-play should be part of the standard recipe.

## Setup

Architecture and PPO settings were held at the current default `mlp` / `sum`
model, 200 updates, and 16000 episodes per update. New runs used seeds 0/1/2.
The old v0.1 random-only seed-0 checkpoint is included as the direct reference
point for the curriculum lift.

| Run | Training curriculum | Seed | Final checkpoint | Trainer eval | Wall time |
|---|---|---:|---|---:|---:|
| A0 random-only reference | `random` | 0 | `models/ppo_agent_0421_1915_upd200_wins3193.pth` | n/a | n/a |
| mixed seed0 | `random,heuristic,simple` | 0 | `models/ppo_agent_0427_1914_upd200_wins3023.pth` | 3023/3200 = 94.5% | 1h54m |
| mixed seed1 | `random,heuristic,simple` | 1 | `models/ppo_agent_0428_0151_upd200_wins3065.pth` | 3065/3200 = 95.8% | 2h29m |
| mixed seed2 | `random,heuristic,simple` | 2 | `models/ppo_agent_0428_0416_upd200_wins3044.pth` | 3044/3200 = 95.1% | 2h25m |
| self-play seed0 | fixed mix + snapshots | 0 | `models/ppo_agent_0427_2321_upd200_wins3038.pth` | 3038/3200 = 94.9% | 4h07m |
| self-play seed1 | fixed mix + snapshots | 1 | `models/ppo_agent_0428_0836_upd200_wins3057.pth` | 3057/3200 = 95.5% | 4h20m |
| self-play seed2 | fixed mix + snapshots | 2 | `models/ppo_agent_0428_1249_upd200_wins3078.pth` | 3078/3200 = 96.2% | 4h13m |

Trainer eval is a quick 3200-game in-training check. The locked gauntlet below
is the primary fixed-opponent metric.

## Locked gauntlet

Each checkpoint was evaluated with `eval --opponents random,heuristic,simple
--games 15000 --seed 0`, giving 5000 games per opponent type.

| Run | vs random | vs heuristic | vs simple | Overall |
|---|---:|---:|---:|---:|
| A0 random-only seed0 | 4994/5000 = 99.88% | 3403/5000 = 68.06% | 4831/5000 = 96.62% | 13228/15000 = 88.19% |
| mixed seed0 | 4999/5000 = 99.98% | 4276/5000 = 85.52% | 4947/5000 = 98.94% | 14222/15000 = 94.81% |
| mixed seed1 | 4996/5000 = 99.92% | 4305/5000 = 86.10% | 4952/5000 = 99.04% | 14253/15000 = 95.02% |
| mixed seed2 | 4997/5000 = 99.94% | 4353/5000 = 87.06% | 4936/5000 = 98.72% | 14286/15000 = 95.24% |
| self-play seed0 | 4997/5000 = 99.94% | 4343/5000 = 86.86% | 4949/5000 = 98.98% | 14289/15000 = 95.26% |
| self-play seed1 | 4996/5000 = 99.92% | 4390/5000 = 87.80% | 4938/5000 = 98.76% | 14324/15000 = 95.49% |
| self-play seed2 | 5000/5000 = 100.00% | 4372/5000 = 87.44% | 4947/5000 = 98.94% | 14319/15000 = 95.46% |

Aggregated across the three new seeds:

| Curriculum | vs random | vs heuristic | vs simple | Overall |
|---|---:|---:|---:|---:|
| mixed fixed | 99.95% | 86.23% | 98.90% | 95.02% |
| fixed + self-play | 99.95% | 87.37% | 98.89% | 95.40% |

The curriculum lift over random-only training is large and stable. Relative to
A0 seed0, the fixed mixed curriculum improves the heuristic matchup by
17.5-19.0 percentage points and the overall gauntlet by 6.6-7.1 percentage
points.

Self-play adds a smaller but consistent gauntlet gain over fixed mixed training:
+1.14pp pooled vs heuristic and +0.38pp overall. Random and simple are already
saturated, so almost all discriminating signal comes from `heuristic`.

## Cross-play Elo

Round-robin tournament of the seven PPO checkpoints plus `random`, `heuristic`,
and `simple`: 45 pairings x 1000 games = 45000 games total.

| Rank | Player | Curriculum | Elo | W / L | Win % |
|---:|---|---|---:|---:|---:|
| 1 | `0428_1249_upd200` | self-play seed2 | 1210 | 6501 / 2499 | 72.2% |
| 2 | `0428_0836_upd200` | self-play seed1 | 1207 | 6472 / 2528 | 71.9% |
| 3 | `0427_2321_upd200` | self-play seed0 | 1192 | 6319 / 2681 | 70.2% |
| 4 | `0428_0151_upd200` | mixed seed1 | 1139 | 5762 / 3238 | 64.0% |
| 5 | `0428_0416_upd200` | mixed seed2 | 1136 | 5730 / 3270 | 63.7% |
| 6 | `0427_1914_upd200` | mixed seed0 | 1132 | 5686 / 3314 | 63.2% |
| 7 | `0421_1915_upd200` | A0 random-only seed0 | 1000 | 4303 / 4697 | 47.8% |
| 8 | `heuristic` | fixed agent | 840 | 2977 / 6023 | 33.1% |
| 9 | `simple` | fixed agent | 451 | 1169 / 7831 | 13.0% |
| 10 | `random` | fixed agent | -15 | 81 / 8919 | 0.9% |

Mean Elo by curriculum:

| Curriculum | Mean Elo | Seed stdev | Delta vs A0 seed0 |
|---|---:|---:|---:|
| A0 random-only seed0 | 1000 | n/a | - |
| mixed fixed | 1136 | 4 | +136 |
| fixed + self-play | 1203 | 10 | +203 |

The Elo result is stronger than the fixed gauntlet result for self-play. All
three self-play seeds rank above all three mixed seeds, and the self-play group
beats the mixed group head-to-head in 5376/9000 games = 59.7%.

Key grouped head-to-head checks:

| Pairing | Row wins |
|---|---:|
| mixed fixed vs A0 random-only seed0 | 2024/3000 = 67.5% |
| fixed + self-play vs A0 random-only seed0 | 2310/3000 = 77.0% |
| fixed + self-play vs mixed fixed | 5376/9000 = 59.7% |

## Findings

1. **The default curriculum change is clearly worth it.** Moving from
   random-only training to the fixed mixed pool produces a 17-19pp lift vs
   `heuristic`, about +6.8pp overall gauntlet, and +136 Elo for the seed-0
   reference comparison.
2. **Self-play is stronger, especially in cross-play.** Its fixed-gauntlet gain
   is modest because the gauntlet is nearly saturated outside `heuristic`, but
   the head-to-head ladder is unambiguous: all self-play seeds rank above all
   mixed-only seeds, with a 59.7% grouped head-to-head score.
3. **Self-play is also much more expensive and more operationally complex.**
   The mixed runs averaged about 2.3h each; self-play averaged about 4.2h each.
   It also introduces snapshot-pool state, resume gotchas, and nonstationary
   training dynamics.
4. **The current fixed gauntlet is starting to saturate.** `random` and
   `simple` no longer separate strong checkpoints. Future curriculum work needs
   either cross-play/Elo, a stronger fixed heuristic, or a harder scripted
   opponent to measure progress cleanly.

## Recommendation

Make **mixed fixed opponents** the v0.2 default and keep **self-play opt-in**.

The fixed mixed curriculum is the right default because it is stable,
reproducible, fast enough for iteration, and fixes the real weakness in the
old random-only baseline. Self-play should be treated as the stronger research
or "best checkpoint" mode rather than the default training contract until it is
tested against a compute-matched mixed baseline.

The best current checkpoint for strength is `models/ppo_agent_0428_1249_upd200_wins3078.pth`
by Elo, with `models/ppo_agent_0428_0836_upd200_wins3057.pth` effectively tied.

## Follow-ups

1. Run a compute-matched control: mixed fixed opponents for roughly the same
   wall-clock budget as self-play, likely 350-400 updates, to determine whether
   self-play is better than simply training longer.
2. Add or design a stronger fixed opponent. The current gauntlet is too close
   to saturated for random/simple and leaves `heuristic` as the only meaningful
   fixed-opponent signal.
3. Keep `--self-play` explicit in CLI examples, especially when resuming a
   checkpoint with a snapshot pool.

## Reproduction

Training logs:

```text
logs/curriculum_default_seed0_20260427_172015.log
logs/curriculum_selfplay_seed0_20260427_172015.log
logs/overnight_mixed_seed1_20260427_2303.log
logs/overnight_mixed_seed2_20260427_2303.log
logs/overnight_selfplay_seed1_20260427_2303.log
logs/overnight_selfplay_seed2_20260427_2303.log
```

Eval logs:

```text
results/curriculum_v02/A0_random_only_seed0.log
results/curriculum_v02/mixed_seed0.log
results/curriculum_v02/mixed_seed1.log
results/curriculum_v02/mixed_seed2.log
results/curriculum_v02/selfplay_seed0.log
results/curriculum_v02/selfplay_seed1.log
results/curriculum_v02/selfplay_seed2.log
results/curriculum_v02/curriculum_elo.log
```

Gauntlet command shape:

```powershell
.venv\Scripts\python.exe -m src eval `
    --model <checkpoint> `
    --opponents random,heuristic,simple `
    --games 15000 `
    --seed 0 `
    --num-workers 16
```

Elo command shape:

```powershell
.venv\Scripts\python.exe -m src elo `
    --checkpoints <A0> <mixed0> <mixed1> <mixed2> <self0> <self1> <self2> `
    --agents random,heuristic,simple `
    --games-per-pair 1000 `
    --num-workers 16
```
