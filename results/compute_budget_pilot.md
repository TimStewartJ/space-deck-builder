# Compute budget pilot and fixed-opponent baseline

**Status:** Complete (2026-08-16); selected **400 updates × 2,000 episodes**
**Evidence:** [`compute_budget_pilot/pilot_20260815_223359/`](compute_budget_pilot/pilot_20260815_223359/)
and [`strength_baseline/pilot_20260815_223359/`](strength_baseline/pilot_20260815_223359/)

> **Seat protocol:** the gauntlets in this writeup predate exact seat
> alternation and randomized seats. The authoritative exact-seat numbers for
> the same three fixed-opponent checkpoints are in
> [`optimized_selfplay_baseline.md`](optimized_selfplay_baseline.md). Logs,
> metrics, manifests, and analysis are committed here; the pilot checkpoints
> are archived off-repo, and the three fixed-opponent finals are published in
> the `v0.3-selfplay-baseline` release bundle as `checkpoints/fixed_seed*.pth`.

## Question

The inherited default was 200 updates × 16,000 episodes: about 2.2M samples
per update, consumed by four epochs of 8,192-sample minibatches while the
policy moved only `kl≈0.003`. The only preserved long run was random-only and
saturated by update 20, so the convergence point against the discriminating
Heuristic opponent was unknown.

## Design

Three arms trained on the vector engine against the fixed
`random,heuristic,simple` pool at an identical 800,000-episode budget, with
seed 0 and `--eval-every 5`. Each arm annealed its cosine learning rate over
its own length, so every arm was a complete run.

| Arm | Updates × episodes | Policy iterations |
|---|---|---:|
| A | 50 × 16,000 | 1× |
| B | 200 × 4,000 | 4× |
| C | 400 × 2,000 | 8× |

The preregistered rule read Heuristic first: choose C, then B, if its pooled
last-three Heuristic rate was within 2.5pp of A and its final rate within
3.5pp.

## Result

| Arm | Final Heuristic | Pooled last 3 | Plateau estimate |
|---|---:|---:|---:|
| A | 86.79% | 86.13% | update 35 |
| B | 87.72% | 87.04% | update 115 |
| C | 84.54% | 85.38% | update 235 |

Arm C was 0.75pp below A on the pooled tail and 2.25pp below on the final
rate, so the rule selected **C**: `updates=400`, `episodes=2000`.

## Three-seed fixed-opponent baseline

Seeds 0, 1, and 2 then trained at the selected budget. Their 15,000-game
gauntlets (randomized seats) were:

| Seed | Random | Heuristic | Simple |
|---:|---:|---:|---:|
| 0 | 99.68% | 86.82% | 97.32% |
| 1 | 99.82% | 86.90% | 97.40% |
| 2 | 99.70% | 87.82% | 97.76% |
| Pooled | 99.73% | **87.18%** | 97.49% |

Relative to the April three-seed fixed-mix baseline this was +0.95pp against
Heuristic and −0.22pp overall. Live cross-play Elo against April checkpoints
was impossible: they use 8 zone embeddings and current checkpoints use 10, and
strict loading correctly rejects cross-encoding inference.

These three checkpoints became the fixed-opponent control for
[`optimized_selfplay_baseline.md`](optimized_selfplay_baseline.md), which
re-evaluated them with exact seats (86.77% ± 0.45pp against Heuristic).

## Decision

Adopt 400 × 2,000 for the mixed fixed-opponent recipe and for snapshot
self-play. Revisit only if the model, optimizer, opponent mix, or rollout
semantics change materially.
