# Batched turn search

**Status:** Complete (2026-09-16)
**Code:** `perf/batched-search` (design in
[`docs/decisions/0011-batched-search-and-count-roots.md`](../docs/decisions/0011-batched-search-and-count-roots.md))
**Evidence:** [`batched_search/`](batched_search/), summarized by
[`batched_search/summarize_batched_search.py`](batched_search/summarize_batched_search.py)

## Question

Can turn-bounded search run fast enough to generate training data by
searching many games' decisions together and building search roots from
public counts instead of deep copies, without changing how strongly it plays?

## Method

- **Hardware and launch.** AMD Radeon RX 9070, torch
  2.11.0+rocm7.13.0a20260408, Windows 11. Every measurement was launched
  through Task Scheduler (outside the agent process tree) with the
  `selfplay_seed0` checkpoint at 64 rollouts per decision and up to 8
  candidates.
- **Decision throughput.** `scripts/bench_turn_search.py --mode decisions`
  times `TurnSearch.choose_many` on greedy self-play positions.
- **Game throughput.** `--mode games` and `python -m src search-eval` play
  complete search-versus-greedy games.
- **Strength.** Old implementation (`bfaa7c9`) and new implementation, both
  against the same checkpoint playing greedily, on identical deals with exact
  seat alternation:
  - the published seed-0 ladder deals;
  - 1,000 fresh deals with master seed 7.
  The two implementations sample different search worlds on the same deals,
  so outcomes are compared deal by deal with exact McNemar tests.
- **Evaluation.** Both implementations searched 150 fixed greedy self-play
  positions on CPU with the top two candidates and 128 rollouts each.

## Throughput

| Implementation | Topology | Result |
|---|---|---|
| Previous | 8 workers, one decision at a time | 1,000 budget-64 games in 1,420 s; 86.5 ms per searched decision |
| New | 1 process, decisions at batch 256 | 494 decisions/s, 2.0 ms per searched decision |
| New | 1 process, decisions at batch 1,024 | 611 decisions/s, 1.6 ms per searched decision |
| New | 1 process, decisions at batch 2,048 | 573 decisions/s, 1.7 ms per searched decision |
| New | 1 process, games at batch 512 | 190 games/min, 387 searched decisions/s |
| New | 1 process, games at batch 1,024 | 235 games/min, 489 searched decisions/s |
| New | 2 workers × 1,024 games | 382 games/min (2,048 games in 322 s) |
| New | 2 workers × 1,024 games | 1,000 budget-64 games in 186 s |

The same 1,000-game workload finished 7.6× faster in wall time, from two
processes instead of eight. Per searched decision the cost fell from about
86 ms to about 1.6 ms.

At batch 1,024 a search call spends 39% of its time in rollout policy calls,
21% in the vector engine, 15% populating rollout rows, 13% building roots,
and 8% in critic calls. Each rollout step still pays a fixed ROCm call
overhead of about 2.6 ms, which larger concurrent batches amortize. Batch
2,048 was slower per decision than 1,024.

Before world dealing was vectorized and digest dtype labels cached, batch 512
measured 156 games/min, and populating rows and building roots took 40 s and
20 s of a 177 s search total.

## Strength

| Deals | Old implementation | New implementation | Paired difference (95% CI) | Discordant deals (new only / old only) | McNemar p |
|---|---:|---:|---|---:|---:|
| Seed 0, published deals | 56.9% | 54.9% | −2.0pp (−3.9 to −0.1) | 39 / 59 | 0.05 |
| Seed 7, fresh deals | 54.3% | 54.7% | +0.4pp (−1.4 to +2.2) | 45 / 41 | 0.75 |
| **Pooled** | 55.6% | 54.8% | **−0.8pp (−2.1 to +0.5)** | 84 / 100 | 0.27 |

The seed-0 greedy control (budget 0) reproduced the published run game by
game: the same winner on all 1,000 deals, 501 searcher wins in both. The
batched policy path, game seeding, and seat alternation are unchanged.

The seed-0 difference did not replicate on fresh deals. The pooled interval
includes zero and excludes a loss larger than about two points.

## Candidate evaluation

On 150 fixed positions, the old and new implementations:

- selected identical candidates with identical rollout allocations in 150/150
  positions;
- produced value gaps between the two candidates that correlate at 0.998;
- differed in mean value gap by +0.0006 (95% CI −0.0003 to +0.0016).

Search worlds differ between the implementations, so values are not bitwise
equal; the decision-relevant comparison is statistically identical.

## Conclusions

- Search is about 50× cheaper per searched decision. It generates about 380
  searched games per minute from two processes, enough for training data at
  scale.
- The batched implementation plays with the same strength as the published
  search. Game outcomes, evaluation of fixed positions, and the control all
  agree within noise.
- The largest remaining costs are per-step rollout policy calls and Python
  root building; further gains need fewer GPU calls per decision or native
  root construction.

## Artifacts

- `bench_*.json` — decision and game benchmark outputs, including runner and
  search phase timings;
- `strength_seed0_budget0_64.json` — new implementation, seed-0 deals,
  budgets 0 and 64;
- `strength_seed7_budget64_old.json`, `strength_seed7_budget64_new.json` —
  both implementations on fresh seed-7 deals;
- `compare_search_values.py`, `values_old.json`, `values_new.json` — fixed
  position evaluation (run the script once per repository checkout);
- `summarize_batched_search.py` — every number in this writeup.
