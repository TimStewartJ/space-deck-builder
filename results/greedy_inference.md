# Greedy inference

**Status:** Protocol-compliant variant (eval protocol v0.1.3)
**Code:** `0d3f8c8` (`eval --temperature`, `elo --checkpoint-temperatures`)
**Checkpoints:** the six `v0.3-selfplay-baseline` finals
([`optimized_selfplay_baseline.md`](optimized_selfplay_baseline.md))
**Evidence:** [`greedy_inference/`](greedy_inference/), reproducible with
[`greedy_inference/run_greedy_inference.ps1`](greedy_inference/run_greedy_inference.ps1)

## Question

Every published result evaluates checkpoints by **sampling** their policy,
and the reference self-play policy still has entropy near 0.72 at the end of
training. Does choosing the highest-probability legal action (temperature 0)
make the same checkpoints stronger?

## Method

- The same six checkpoints: three self-play seeds and three fixed-opponent
  seeds, each trained for 400 updates × 2,000 episodes.
- Locked gauntlet at temperature 0: 5,000 games against each fixed opponent
  per checkpoint, 2,500 in each seat, seed = training seed. Temperature-1
  numbers are the published exact-seat gauntlets from the same checkpoints.
- One exact-seat tournament with every checkpoint at temperatures 1 and 0
  plus the fixed agents: 15 participants, 105 pairings × 1,000 games, anchor
  `fixed_seed0` (temperature 1) = 1000 to match the published scale.
- Intervals are Student-t 95% intervals across the three training seeds
  (df = 2). Deltas are paired by seed.
- AMD Radeon RX 9070, torch 2.11.0+rocm7.13.0a20260408, Windows, launched
  through Task Scheduler.

## Gauntlet

### Variant: self-play, temperature 0

|                  | vs. random      | vs. heuristic   | vs. simple      | overall         |
|------------------|-----------------|-----------------|-----------------|-----------------|
| Temperature 1    | 99.79% ± 0.15   | 86.36% ± 2.03   | 97.06% ± 1.04   | 94.40% ± 0.74   |
| Temperature 0    | 99.94% ± 0.05   | **92.51% ± 0.87** | 99.11% ± 0.46 | 97.19% ± 0.33   |
| Paired delta     | +0.15pp ± 0.13  | **+6.15pp ± 1.28** | +2.05pp ± 0.65 | +2.78pp ± 0.45 |
| Per-seed Heuristic (T=0) | seed0=92.86%  seed1=92.16%  seed2=92.50% | | | |

### Variant: fixed-opponent control, temperature 0

|                  | vs. random      | vs. heuristic   | vs. simple      | overall         |
|------------------|-----------------|-----------------|-----------------|-----------------|
| Temperature 1    | 99.79% ± 0.08   | 86.77% ± 0.45   | 97.55% ± 0.63   | 94.71% ± 0.10   |
| Temperature 0    | 99.97% ± 0.08   | **92.95% ± 0.15** | 99.23% ± 0.45 | 97.38% ± 0.21   |
| Paired delta     | +0.17pp ± 0.15  | **+6.18pp ± 0.55** | +1.68pp ± 0.33 | +2.68pp ± 0.24 |
| Per-seed Heuristic (T=0) | seed0=92.90%  seed1=93.02%  seed2=92.94% | | | |

Greedy decoding removes about 45% of losses against Heuristic and about half
of all gauntlet losses in both leagues. Every paired interval excludes zero.
Every per-opponent result has exactly 2,500 games in each seat.

## Cross-play Elo

| Rank | Participant | Elo |
|---:|---|---:|
| 1 | selfplay_seed0@T0 | 1190 |
| 2 | selfplay_seed2@T0 | 1184 |
| 3 | selfplay_seed1@T0 | 1152 |
| 4 | selfplay_seed0 | 1148 |
| 5 | selfplay_seed2 | 1141 |
| 6 | selfplay_seed1 | 1107 |
| 7 | fixed_seed0@T0 | 1074 |
| 8 | fixed_seed2@T0 | 1071 |
| 9 | fixed_seed1@T0 | 1070 |
| 10 | fixed_seed1 | 1003 |
| 11 | fixed_seed0 | 1000 |
| 12 | fixed_seed2 | 997 |
| 13 | heuristic | 675 |
| 14 | simple | 367 |
| 15 | random | −99 |

Head-to-head, each checkpoint at temperature 0 beats itself at temperature 1:

| Checkpoint | Greedy vs sampled |
|---|---:|
| selfplay_seed0 | 54.3% |
| selfplay_seed1 | 55.4% |
| selfplay_seed2 | 55.4% |
| fixed_seed0 | 62.6% |
| fixed_seed1 | 62.9% |
| fixed_seed2 | 59.4% |

- Greedy self-play gains **+43 Elo** on average and greedy fixed-opponent
  checkpoints gain **+72 Elo** over their sampled selves.
- The league ordering is unchanged: greedy self-play checkpoints win 64.6% of
  their games against greedy fixed-opponent checkpoints (68.2% when both
  sample).
- Heuristic sits 325 Elo below the anchor here versus 275 in the published
  tournament. That gap is not a shift in Heuristic's strength: MLE ratings
  depend on the participant set, and adding six stronger greedy participants
  moves every rating. Compare ratings only within one tournament.

## Conclusions

- Published strength numbers understate what the trained policies can do.
  The locked protocol measures sampled play and remains valid for comparing
  variants at the same temperature.
- The right inference-time baseline for search is the same checkpoint at
  temperature 0; search must beat that player, not sampled play.
- Temperature is a separate variant: these results do not replace the
  temperature-1 results, and training rollouts still sample.

## Artifacts

- `<checkpoint>_T0.json` / `.log` — locked exact-seat gauntlets with provenance;
- `elo.json` — participants, temperatures, MLE ratings, and all 105 pairings;
- `elo.log` — the tournament log and leaderboard.
