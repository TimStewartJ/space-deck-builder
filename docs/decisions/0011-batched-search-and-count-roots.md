# 0011 - Batched turn search and count-based roots

**Status:** Accepted on `perf/batched-search` (2026-09-16)

## Context

Turn-bounded search ([0010](0010-turn-bounded-search.md)) cost about 80 ms per
searched decision at 64 rollouts in one GPU worker:

- about 55 ms of policy and critic calls. Each call carried a batch of at most
  a few hundred rows and a fixed ROCm launch and synchronization overhead of
  about 2.5 ms, regardless of batch size;
- about 17 ms projecting the information set through deep copies of the game;
- the rest in vector engine work.

Cost scaled with rollout steps, not rollouts: 16, 64, and 256 rollouts measured
84, 96, and 121 ms per decision. Eight worker processes produced about 83
searched decisions per second, far too few to generate search targets for
training.

## Decision

**Count-based roots.** `SearchRootBuilder` writes the acting player's vector
root straight from the live game without copying it or reading hidden zones:

- visible zones, resources, and pending choices are written directly;
- the acting player's deck is public ownership minus visible zones;
- the opponent's unseen cards are public ownership minus visible zones, held
  as publicly known hand cards plus the information set's canonical allocation;
- the trade deck is the public trade supply minus the trade row.

`from_games` and the root builder share one row writer. The root builder reuses
the information-set validators. A SHA-256 digest of the root row's public
fields, viewer, and known hand cards seeds the search.

**Batched search.** `TurnSearch.choose_many` searches many games' decisions
together:

- every decision keeps its own sequential-halving plan and budget;
- each round gathers every surviving candidate × world row of every decision
  into one vector population;
- every rollout step makes one actor-only evaluation for all active rows and
  transfers only argmax actions;
- leaf and capped values use one critic evaluation per round.

`choose(game)` is `choose_many([game])`.

**Vectorized dealing.** World hands come from a counter-based SplitMix64 hash
of the search seed, world id, and card position. Sorting the keys deals a
uniform subset of the unseen multiset. Every world of every decision in a round
is dealt by a few array operations, and a world's deal does not depend on
which other worlds share the population.

**Batched games.** `BatchedGameRunner` keeps up to `batch_size` games in
flight:

- scalar agents act immediately;
- waiting policy decisions share one forward pass;
- waiting search decisions share one `choose_many` call;
- sampled policy seats draw from a per-game generator.

`python -m src search-eval` uses the runner (`--batch-size`).

## Rejected alternatives

**Deep-copy projection for seeding.** Keeping `project_information_set` on the
hot path only to reproduce its fingerprint costs about 17 ms per decision.

**CPU inference for small tail batches.** Late rollout steps often have few
active rows, where CPU inference is faster than a GPU call. Mixing devices
would make a row's greedy action depend on how many rows share its step,
because CPU and GPU arithmetic differ, so outcomes would depend on batch size.

**CUDA graphs or TorchInductor.** The ROCm stack probe recompiled for variable
batch sizes and gave no warm speedup
([`results/rocm715_stack_probe.md`](../../results/rocm715_stack_probe.md)).

## Consequences

Measured through Task Scheduler on the RX 9070 at 64 rollouts with four Numba
threads, using greedy self-play positions (decision mode) and complete games
against the greedy policy (game mode):

| Mode | Batch | Throughput | Per searched decision |
|---|---:|---:|---:|
| Previous search, 8 workers, games | — | 1,000 budget-64 games in 1,420 s | 86.5 ms |
| Batched, 2 workers, same games | 1024 each | 1,000 budget-64 games in 186 s | 2.6 ms |
| Decisions, 1 process | 256 | 494 decisions/s | 2.0 ms |
| Decisions, 1 process | 1024 | 611 decisions/s | 1.6 ms |
| Games, 1 process | 512 | 190 games/min, 387 searched decisions/s | — |
| Games, 1 process | 1024 | 235 games/min, 489 searched decisions/s | — |
| Games, 2 workers | 1024 each | 382 games/min | — |

On identical deals the batched search takes 34× less time per searched
decision, 30× less process time, and 7.6× less wall time. The decision
benchmark's 1.6 ms is a single-process figure without GPU sharing.

At batch 1024 a search call spends 39% of its time in rollout policy calls, 21%
in the vector engine, 15% populating rows, 13% building roots, and 8% in critic
calls. Batch 2048 was slower than 1024 per decision.

Search worlds are seeded from the root digest and dealt by the hash deal, so
the same seeds sample different worlds than the implementation that produced
[`results/search_uplift.md`](../../results/search_uplift.md). Reproducing those
exact games requires commit `1849b19`. Against the previous implementation on
2,000 paired deals the batched search shows no detectable strength difference
(−0.8pp, 95% CI −2.1 to +0.5), and the greedy control reproduced all 1,000
published control games
([`results/batched_search.md`](../../results/batched_search.md)).

## Verification

- roots equal the vector conversion of `project_information_set(...)` on more
  than 300 positions across eight complete games, and with publicly known
  hand, known top-deck, pending copy, and free-acquisition states;
- roots and digests are identical for games that differ only in hidden
  information, the builder makes no deep copies, and it rejects the same
  invalid states as the projection;
- `choose_many` equals individual `choose` exactly with a scripted policy and
  within 1e-9 with the checkpoint in float64, including per-decision budgets;
- dealing many blocks together equals dealing each alone, and dealt hands
  match the expected per-card frequencies within five standard errors over
  4,000 worlds;
- the batched arena reproduces games played by scalar search and policy agents
  on identical deals against greedy and Heuristic baselines, and runner outcomes
  do not change between batch sizes 1 and 4 for search, greedy, sampled, and
  Heuristic seats;
- on 150 fixed positions the previous and batched implementations select
  identical candidates with identical rollout allocations, and their value
  gaps correlate at 0.998.
