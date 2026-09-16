# Expert iteration round 1

**Status:** Complete (2026-09-16). **Pre-registered outcome: FAIL**, by two
games on the primary gate.
**Design:** [`docs/decisions/0012-expert-iteration-from-turn-search.md`](../docs/decisions/0012-expert-iteration-from-turn-search.md),
committed before any data was generated (`612b5ac`)
**Evidence:** [`expert_iteration/round1/`](expert_iteration/round1/), produced by
[`expert_iteration/run_round1.ps1`](expert_iteration/run_round1.ps1) and
summarized by
[`expert_iteration/summarize_round1.py`](expert_iteration/summarize_round1.py)

## Question

Does training `selfplay_seed0` on its own search's improved policies and game
outcomes produce a policy that plays better without search?

## Method

The pre-registered round ran through Task Scheduler from a worktree pinned to
`612b5ac`. It used the RX 9070 and torch 2.11.0+rocm7.13.0a20260408.

1. **Self-play.** 8,192 search-versus-search games with `selfplay_seed0` at
   budget 64 and 8 candidates, seed 0, two workers × 1,024 concurrent games.
   The run produced 2,230,469 positions in 44.4 minutes.
2. **Distillation.** Masked cross-entropy to the completed-Q improved policy
   plus 0.5 × outcome squared error. Adam at lr 1e-4, batch 8,192, at most 8
   epochs, patience 2, gradient clip 0.5. Games with index % 20 == 0 were held
   out (113,760 validation positions). It took 72 seconds.
3. **Primary gate.** `elo` between the distilled and source checkpoints, both
   at temperature 0: 2,000 games with exact seat alternation. Pass requires a
   score of at least 52.9% (+20 Elo) and a Wilson 95% lower bound above 50%.
4. **Retention gate.** Locked 15,000-game gauntlet at temperature 0, seed 0.
   Pass requires at least 91.86% against Heuristic.
5. **Diagnostic, not gating.** `search-eval` on the distilled checkpoint at
   budgets 0 and 64 on 1,000 deals.

## Distillation

| Epoch | Policy KL to target | Search-move agreement | Initial greedy agreement | KL from initial | Value MSE | Value explained variance |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (source) | 0.0900 | 97.17% | 100.00% | 0.0000 | 0.6067 | 0.395 |
| 1 | 0.0772 | 93.55% | 95.34% | 0.0092 | 0.5731 | 0.425 |
| 3 | 0.0752 | 93.51% | 95.18% | 0.0119 | 0.5706 | 0.427 |
| **5 (selected)** | **0.0745** | **93.26%** | **94.89%** | **0.0125** | **0.5707** | **0.426** |
| 7 | 0.0741 | 92.98% | 94.56% | 0.0130 | 0.5729 | 0.424 |

Validation loss was best at epoch 5; epochs 6 and 7 did not improve it.

- **Policy.** The student closed about 17% of the gap to the search targets
  (KL 0.090 → 0.0745 nats), and its critic explains more outcome variance
  (0.395 → 0.426).
- **Agreement.** The student's greedy move agrees *less* often with search's
  move (97.2% → 93.3%). The completed-Q targets are soft (entropy 0.547 nats),
  so when two actions have similar targets the student's argmax follows the
  smaller logit differences. The student changes about 5% of the source's
  greedy decisions, not only the ~3% search changes.

## Gates

| Gate | Result | Threshold | Outcome |
|---|---|---|---|
| Primary: distilled vs source at T=0 | 1,056/2,000 = **52.8%** (Wilson 95% CI 50.6–55.0%); **+19 Elo** (+4 to +35) | ≥ 52.9% and lower bound > 50% | **FAIL** |
| Retention: Heuristic, locked T=0 gauntlet | 4,632/5,000 = **92.64%** | ≥ 91.86% | PASS |

The distilled checkpoint is stronger than its source. An exact binomial test
gives two-sided p = 0.013, and the interval excludes zero. Its score is two
games below the pre-registered +20 Elo effect threshold, so round 1 fails by
the rule fixed before the experiment. The thresholds are not revised after
seeing the result.

The full temperature-0 gauntlet was:

| Opponent | Distilled (this round) | Source, published T=0 |
|---|---:|---:|
| Random | 100.00% | 99.92% |
| Heuristic | 92.64% | 92.86% |
| Simple | 99.10% | 99.24% |

Every opponent had exactly 2,500 games in each seat. The fixed gauntlet is
saturated, and differences of this size are within its noise.

## Diagnostic: search on the distilled checkpoint

| Budget | Search wins | Win rate (95% CI) | Elo vs distilled greedy |
|---:|---:|---:|---:|
| 0 (control) | 492/1,000 | 49.2% (46.1–52.3%) | −6 |
| 64 | 579/1,000 | 57.9% (54.8–60.9%) | +55 |

Search still improves on the distilled checkpoint's own greedy play (+55 Elo,
changing 3.1% of decisions). That is at least as much as on the source
checkpoint (+44 Elo pooled over three seeds): distillation did not use up the
advantage search adds. These are separate matchups, so the Elo figures need
not add.

## Conclusions

- One round of expert iteration made the greedy policy measurably stronger
  (+19 Elo, 95% CI +4 to +35) with no detectable loss against the fixed
  gauntlet, but the gain missed the pre-registered +20 Elo threshold, so the
  round fails.
- Search adds as much on top of the distilled policy as it did on the source,
  so further rounds are not ruled out by a shrinking search advantage.
- The distilled policy changes more decisions than search does, and its
  agreement with search's moves fell. The soft completed-Q targets transfer
  search's preferences only partly.

The pre-registered consequence of a failed gate is to improve leaf evaluation
before running more rounds.

## Artifacts

- `selfplay_summary.json`, `selfplay.log` — generation identity and progress;
- `distill_history.json`, `distill.log` — per-epoch training and validation
  metrics;
- `gate_elo_T0.json`, `gate_gauntlet_T0.json` and logs — both gates;
- `diagnostic_search_eval.json` and log — the diagnostic match.

The distilled checkpoint
(`distilled_seed0.pth`, SHA-256 `af1cbea817b23d63dd7d89feb2c2de3d7922006ac11b7da4bef1ef93e93b60ca`),
the 35 self-play shards, and the diagnostic games journal are archived off-repo
under `D:\space-deck-builder-archive\expert-iteration-round1-20260916`. The
checkpoint has not been published because nothing has been pushed; publish it
as a release asset when the branch is pushed.
