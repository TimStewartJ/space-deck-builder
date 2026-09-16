# Turn-bounded search uplift

**Status:** Complete (2026-09-16)
**Code:** `1849b19` on `feat/turn-search` (`python -m src search-eval`; design in
[`docs/decisions/0010-turn-bounded-search.md`](../docs/decisions/0010-turn-bounded-search.md))
**Checkpoints:** `selfplay_seed0/1/2` from the `v0.3-selfplay-baseline`
release ([`optimized_selfplay_baseline.md`](optimized_selfplay_baseline.md))
**Evidence:** [`search_uplift/`](search_uplift/), reproducible with
[`search_uplift/run_search_uplift.ps1`](search_uplift/run_search_uplift.ps1)
and summarized by
[`search_uplift/summarize_search_uplift.py`](search_uplift/summarize_search_uplift.py)

## Question

Does information-safe, turn-bounded search guided by a checkpoint's own policy
and critic beat that checkpoint's strongest direct play: choosing its
highest-probability action (see [`greedy_inference.md`](greedy_inference.md))?
How does strength change with the rollout budget?

## Method

- **Match:** every game pits the searcher against the same checkpoint at
  temperature 0 (greedy). Neither player sees hidden information.
- **Budgets:** rollouts per decision; a decision with one legal action after
  END_TURN suppression is played without search. Each budget compares at
  most 8 highest-prior candidates by sequential halving. Budget 0 plays
  greedily and is the fairness control.
- **Pairing and seats:** game `index` uses the same deal RNG for every budget,
  and the searcher moves first on even indices, so every budget gets exactly
  half the first seats on identical deals. Paired tests compare arms deal by
  deal (exact McNemar).
- **Pre-registered design:** budgets 0/16/64/256 × 1,000 deals on
  `selfplay_seed0`, then budgets 0/64 × 500 deals on `selfplay_seed1` and
  `selfplay_seed2` as replications.
- **Hardware:** AMD Radeon RX 9070, torch 2.11.0+rocm7.13.0a20260408, Windows
  11. Eight worker processes shared the GPU, launched through Task Scheduler.
  A Windows Update restart at 02:30 interrupted the seed-0 ladder after 1,778
  games. The run resumed from its games journal: the lost in-flight games were
  replayed, and every (budget, deal) appears exactly once in the evidence.

This is a head-to-head match, not the locked fixed-opponent gauntlet. Elo is
the logistic transform of the searcher's score against the greedy checkpoint,
with Wilson 95% intervals. SPRT reports the log-likelihood ratio for +20 Elo
against 0 (accept at ≥ +2.94, reject at ≤ −2.94 for α = β = 0.05).

## Budget ladder: selfplay_seed0

| Budget | Search wins | Win rate (95% CI) | Elo vs greedy (95% CI) | SPRT LLR | First seat | Second seat | Decisions changed | Mean search |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 501/1000 | 50.1% (47.0–53.2) | +1 (−21 to +22) | −1.5 | 56.2% | 44.0% | — | — |
| 16 | 548/1000 | 54.8% (51.7–57.9) | +33 (+12 to +55) | +3.9 | 58.8% | 50.8% | 2.7% | 84 ms |
| 64 | 569/1000 | **56.9% (53.8–59.9)** | **+48 (+27 to +70)** | +6.3 | 59.8% | 54.0% | 2.8% | 96 ms |
| 256 | 567/1000 | 56.7% (53.6–59.7) | +47 (+25 to +69) | +6.1 | 58.4% | 55.0% | 3.8% | 121 ms |

Paired on identical deals:

- budget 16 vs 0: 136 deals won only with search, 89 only without (p = 0.0021);
- budget 64 vs 0: 154 vs 86 (p = 1.3 × 10⁻⁵);
- budget 256 vs 0: 184 vs 118 (p = 0.00017);
- budget 64 vs 16: 73 vs 52 (p = 0.073);
- budget 256 vs 64: 85 vs 87 (p = 0.94).

## Replications at budget 64

| Checkpoint | Deals | Control (budget 0) | Search (budget 64) | Elo vs greedy (95% CI) | Paired deals (search only / control only) | McNemar p |
|---|---:|---:|---:|---|---:|---:|
| selfplay_seed0 | 1000 | 50.1% | 56.9% | +48 (+27 to +70) | 154 / 86 | 1.3 × 10⁻⁵ |
| selfplay_seed1 | 500 | 51.4% | 55.4% | +38 (+7 to +68) | 69 / 49 | 0.080 |
| selfplay_seed2 | 500 | 46.2% | 56.2% | +43 (+13 to +74) | 98 / 48 | 4.3 × 10⁻⁵ |
| **Pooled** | 2000 | 49.5% | **56.4% (54.2–58.5)** | **+44 (+29 to +60)** | 321 / 183 | 8.2 × 10⁻¹⁰ |

All three controls are consistent with 50%, as they must be with the same
greedy policy in both seats. Every searched arm is above 50%, and the pooled
SPRT log-likelihood ratio at budget 64 is +11.3.

Mean search time is wall time per searched decision with eight workers sharing
one GPU. It depends on host load, not on outcomes, because search is budgeted
by rollouts rather than time. Median time at budget 64 was about 78 ms for
seeds 0 and 1. Seed 2 also started near 78 ms, then rose to about 200–230 ms
after its first quarter of games (per-game medians in completion order, from
the archived games journal). Host load was not recorded.

## Conclusions

- **Search improves play.** At 64 rollouts per decision the searcher beats
  its own greedy policy on 56.4% of 2,000 paired deals (+44 Elo) across three
  independent checkpoints. It changes only about 3% of decisions.
- **No detectable gain beyond 64 rollouts.** On `selfplay_seed0`, 256
  rollouts scored the same as 64 (85 vs 87 discordant deals; −0.2pp, paired
  95% CI −2.8 to +2.4pp, about ±18 Elo) even though search changed more
  decisions (3.8% vs 2.8%). The 16-to-64 step (+15 Elo, p = 0.073) fits
  within that interval, and no pair of searched budgets differs
  significantly. The ladder is consistent with saturation but does not show
  where it begins.
- **Leaf evaluation is a plausible limit.** More rollouts reduce sampling
  noise but not the systematic error of greedy continuations and the critic
  leaf value. A flat ladder is consistent with that bias; these experiments do
  not isolate it.
- **Search and greedy decoding were measured separately.** Head-to-head,
  greedy decoding beat sampled play for these checkpoints in 54.3–55.4% of
  games (about +35 Elo on average; [`greedy_inference.md`](greedy_inference.md)),
  and search beats greedy play by +44 Elo. Search against sampled play was
  not measured, and Elo gains from different matchups need not add.

## Next steps

1. Improve leaf values before spending more rollouts, since 256 rollouts
   showed no detectable gain over 64: evaluate deeper (the opponent's turn
   with the policy), or average the critic over several opponent hand
   re-deals at the leaf.
2. Use search targets for training (expert iteration): record searched visit
   distributions and values, distill them into the policy, and verify with
   paired matches.
3. Cut per-decision cost before large-scale data generation. Projection
   (~17 ms of deep copies) and per-step GPU policy calls (~50 ms in total per
   decision at budget 64) dominate. Batching many games' searches into one
   vector population would remove most of that overhead.

## Artifacts

- `selfplay_*.json` — configuration, per-budget summaries, and every game
  (budget, deal index, seat, winner, turns, decisions, search statistics);
- `selfplay_*.log` — orchestrator progress and summary lines;
- `run_search_uplift.ps1` — the pre-registered orchestration;
- `summarize_search_uplift.py` — integrity checks, paired tests, and pooled
  statistics used in this writeup.

The resumable games journals (`*.games.jsonl`) duplicate the per-game rows in
the JSON files and are archived off-repo.
